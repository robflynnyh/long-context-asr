import lcasr
import torch, numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
from lming.decoding import beam_search

from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

from lming.utils import general


from wer import word_error_rate_detail 
from postprocess import post_process
import re
import pickle as pkl

from postprocess import post_process
from functools import partial
import random
import ray

def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines


def log(args, wer, beam_width, alpha, beta, cur_prune_less_than_val, top_am, max_len):
    log_path = args.log_path
    with open(log_path, 'a') as f:
        f.write(f'{args.checkpoint} - wer:{wer} b:{beam_width} a:{alpha} b:{beta} p:{cur_prune_less_than_val} amt:{top_am} ml:{max_len}\n')

@ray.remote(num_gpus=0.1, num_cpus=0.1)
def run_search(
        beam_search_fn, 
        logits, 
        gold_text,
        alpha,
        beta,
        prune_less_than_val,
        top_am,
    ):
    beam_search = beam_search_fn(
        log_probs = logits,
    )
    # print beam search args
    beam_search.run_search(use_tqdm=True)
    text_out = normalize(beam_search.return_text(idx = 0)).lower()
    gold_text = normalize(gold_text).lower()
    return text_out, gold_text

def main(args):    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint['model'] = general.convert_from_ddp(checkpoint['model'])
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

    
    gold_text = all_logits[0]['gold']
    #print(gold_text)
    hyps = []
    golds = []

  
    alpha_range = [0.24, 0.26] # between 
    beta_range = [0.64, 0.66] # between
    beams = [25] # either 
    prune_less_than_vals = [2.0]  # either
    top_am_thresh = [-6] # either

    beam_search_fn = partial(
        beam_search.BeamSearch,
        language_model = language_model,
        tokenizer = tokenizer,
        beam_width = args.beam_width,
        blank_id = tokenizer.vocab_size(),
        alpha = 0.25,
        beta = 0.65,
        debug = False,
        prune_less_than_val = 2.0,
        top_am_threshold = -6,
        max_cache_length = args.max_len,
    )
    beam_search_fn = ray.put(beam_search_fn)
    random.seed(123456)

    for i in range(2):
        cur_alpha = np.random.uniform(alpha_range[0], alpha_range[1])
        cur_beta = np.random.uniform(beta_range[0], beta_range[1])
        cur_beam = np.random.choice(beams)
        cur_prune_less_than_val = np.random.choice(prune_less_than_vals)
        cur_top_am_thresh = np.random.choice(top_am_thresh)


        outputs = []
        for i in range(len(all_logits)):
            logits = all_logits[i]['logits']
            gold_text = all_logits[i]['gold']
            outputs.append(run_search.remote(
                beam_search_fn = beam_search_fn,
                logits = logits,
                gold_text = gold_text,
                alpha = cur_alpha,
                beta = cur_beta,
                prune_less_than_val = cur_prune_less_than_val,
                top_am = cur_top_am_thresh,
            ))
        outputs = ray.get(outputs)
        hyps, golds = zip(*outputs)
        hyps, golds = list(hyps), list(golds)

        
        wer = word_error_rate_detail(hypotheses=hyps, references=golds)[0]
        print(wer, cur_beam, cur_alpha, cur_beta)
        log(
            args = args,
            wer = wer,
            beam_width = cur_beam,
            alpha = cur_alpha,
            beta = cur_beta,
            cur_prune_less_than_val = cur_prune_less_than_val,
            top_am = cur_top_am_thresh,
            max_len = args.max_len,
        )
        #print('0.08997, 0.05626')

        
    #wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    #print(f'WER: {wer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='/mnt/parscratch/users/acp21rjf/language_modelling/spotipile/512_1280/step_540012.pt', help='path to checkpoint')
    
    parser.add_argument('-beams', '--beam_width', type=int, default=1, help='beam width for decoding')
    parser.add_argument('-logits', '--logits_path', type=str, default='./logits/512_dev_logits.pkl', help='path to logits')
    parser.add_argument('-log', '--log_path', type=str, default='tlm_beam.log', help='path to log file')
    parser.add_argument('-max_len', '--max_len', type=int, default=2048, help='max length of text')


    # wer:0.06839218375042852 b:10 a:0.25131218509148656 b:0.8917145455147044
    # wer:0.06804936578676722 b:5 a:0.07152317088066645 b:0.9052812115538087

    args = parser.parse_args()
    main(args)
    
#0.10381063195403971
