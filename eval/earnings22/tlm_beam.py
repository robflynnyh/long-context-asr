import torch, argparse, lcasr, wandb, re, pickle as pkl, ray, random
from tqdm import tqdm
from typing import List 
from omegaconf.omegaconf import OmegaConf
from lming.decoding import beam_search
from lming.utils import general
from lcasr.eval.wer import word_error_rate_detail 
from functools import partial
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

#train_path = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/train/stm/EllenGustafson_2010X.stm'
train_path = '/mnt/parscratch/users/acp21rjf/spotify/txt/spotify-podcasts-2020/podcasts-transcripts/0/0/show_00BnuPjwbyMPxVIM7NimQj/7sHyO8wLeEd1LuxfS8AIls.json'


def log(args, wer, beam_width, alpha, beta):
    log_path = args.log_path
    with open(log_path, 'a') as f:
        f.write(f'{args.checkpoint} - wer:{wer} b:{beam_width} a:{alpha} b:{beta}\n')

@ray.remote(num_gpus=0.0, num_cpus=1.0)
def run_search(beam_search_fn, logits, gold_text):
    beam_search = beam_search_fn(log_probs = logits)
    beam_search.run_search(use_tqdm=True)
    text_out = normalize(beam_search.return_text(idx = 0)).lower()
    gold_text = normalize(gold_text).lower()
    return text_out, gold_text

@ray.remote(num_gpus=0.15, num_cpus=0.2)
def run_search_gpu(beam_search_fn, logits, gold_text):
    beam_search = beam_search_fn(log_probs = logits)
    beam_search.run_search(use_tqdm=True)
    text_out = normalize(beam_search.return_text(idx = 0)).lower()
    gold_text = normalize(gold_text).lower()
    return text_out, gold_text

@torch.no_grad()
def get_init_seq(args:argparse.Namespace, model, tokenizer):
    #text, _ = proc_stm_and_timings(train_path)
    import json
    with open(train_path, 'r') as f:
        txt = json.load(f)
    text = " ".join([el['word'] for el in txt['results'][-1]['alternatives'][0]['words']])

    
    text = text + '. ' if not text.endswith('.') else text + ' '
    tokenized_text = tokenizer.encode(text + '. ') # add a period to the end
    bos = tokenizer.bos_id()
    tokenized_text = [bos] + tokenized_text
    seq_len = 1
    cache_len = args.max_len 

    print(f'Processing with seq_len: {seq_len} and cache_len: {cache_len}')

    all_logits = []
    # process text in chunks of seq_len
    prev_cache = None
    pbar = tqdm(range(0, len(tokenized_text), seq_len), total=len(tokenized_text)//seq_len)
    for i in pbar:
        cur_chunk = tokenized_text[i:i+seq_len]
        cur_chunk = torch.LongTensor(cur_chunk).unsqueeze(0).to(model.device)
   
        logits, _, cached_kvs = model(x = cur_chunk, cache = prev_cache)
        all_logits.append(logits)
        if cache_len != 0:
            prev_cache = cached_kvs
            prev_cache['cache'] = prev_cache['cache'][:, :, :, :, -cache_len:]
            prev_cache['cache_lengths'] = prev_cache['cache_lengths'] * 0 + prev_cache['cache'].shape[-2]
    pbar.close()

    return prev_cache

def main(args):
    wandb.init()
    ray.init(num_cpus=12, num_gpus=0 if not args.use_gpu else 1)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint['model'] = general.convert_from_ddp(checkpoint['model'])

    max_len = args.max_len
  
    model_config = checkpoint['config']
    args.config = model_config
    
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()

    model = general.load_model(config=model_config, vocab_size=tokenizer.vocab_size())
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()



    language_model = beam_search.LanguageModel(
        model = model,
        bos_id = tokenizer.bos_id(),
        device = device,
    )
    
    with open(args.logits_path, 'rb') as f:
        all_logits = pkl.load(f)

    init_cache = get_init_seq(args, model, tokenizer) if args.use_init_cache else None
    print(f"Initial cache/prompt: {init_cache['cache'].shape}") if init_cache else print("No initial cache/prompt")
    gold_text = all_logits[0]['gold']
    #print(gold_text)
    hyps = []
    golds = []

    beam_search_fn = partial(
        beam_search.BeamSearch,
        language_model = language_model,
        tokenizer = tokenizer,
        beam_width = args.beam_width,
        blank_id = tokenizer.vocab_size(),
        alpha = args.alpha,
        beta = args.beta,
        debug = False,
        prune_less_than_val = args.p,
        top_am_threshold = -6,
        max_cache_length = max_len,
        cache_init = init_cache,
    )
    beam_search_fn = ray.put(beam_search_fn)
    random.seed(123456)

    if not args.use_gpu:
        outputs = [
            run_search.remote(
                beam_search_fn = beam_search_fn,
                logits = all_logits[i]['logits'],
                gold_text = all_logits[i]['gold']
            ) for i in range(len(all_logits))
        ]
    else:
        outputs = [
            run_search_gpu.remote(
                beam_search_fn = beam_search_fn,
                logits = all_logits[i]['logits'],
                gold_text = all_logits[i]['gold']
            ) for i in range(len(all_logits))
        ]

    outputs = ray.get(outputs)
    hyps, golds = zip(*outputs)
    hyps, golds = list(hyps), list(golds)    
    print(hyps[0])

    wer = word_error_rate_detail(hypotheses=hyps, references=golds)[0]
    print(f'WER: {wer}')
    wandb.log({"wer": wer})

    if args.log_path != "":
        log(args, wer, args.beam_width, alpha=args.alpha, beta=args.beta)

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='/mnt/parscratch/users/acp21rjf/language_modelling/spotipile/512_1280/step_540012.pt', help='path to checkpoint')
    parser.add_argument('-gpu', '--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('-beams', '--beam_width', type=int, default=25, help='beam width for decoding')
    parser.add_argument('-logits', '--logits_path', type=str, default='/mnt/parscratch/users/acp21rjf/spotify/logits/earnings/n_seq_sched_8192_rp_1_dev.pt', help='path to logits')
    parser.add_argument('-log', '--log_path', type=str, default='', help='path to log file')
    parser.add_argument('-max_len', '--max_len', type=int, default=1024, help='max sequence length')
    parser.add_argument('-alpha', '--alpha', type=float, default=0.42, help='alpha for beam search')
    parser.add_argument('-beta', '--beta', type=float, default=1.95, help='beta for beam search')
    parser.add_argument('-p', '--p', type=float, default=2.96, help='p for beam search')
    parser.add_argument('-dont_use_init_cache', '--dont_use_init_cache', action='store_true', help='dont use init cache')


    args = parser.parse_args()
    args.use_init_cache = not args.dont_use_init_cache
    main(args)
    


# 80s python tlm_beam.py -p 3.25 -beta 2.96 -alpha 0.54 -max_len 1024