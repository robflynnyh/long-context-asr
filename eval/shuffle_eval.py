import torch, argparse, lcasr
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.shuffled_eval import fetch_logits as shuffled_eval
from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from lcasr.decoding.greedy import GreedyCTCDecoder
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
from tqdm import tqdm
import math

from earnings22_full.run import get_text_and_audio as get_text_and_audio_earnings22_full
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16
from this_american_life.run import get_text_and_audio as get_text_and_audio_this_american_life
from spotify.run import get_text_and_audio as get_text_and_audio_spotify

datasets_functions = {
    'tedlium': get_text_and_audio_tedlium,
    'earnings22_full': get_text_and_audio_earnings22_full,
    'earnings22': get_text_and_audio_earnings22,
    'rev16': get_text_and_audio_rev16,
    'this_american_life': get_text_and_audio_this_american_life,
    'spotify': get_text_and_audio_spotify,
    'no_context': lambda split: [None] # dummy function to simplify code
}


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config

    assert torch.cuda.is_available(), 'this eval is written for windowed attention which currently needs flash attention on GPU'

    if args.__dict__.get('disable_flash_attention', False): 
        assert 1==0, 'Flash attention is required for this evaluation'
        args.config.model.flash_attn = False
    
    seq_len = args.seq_len
    subsample_factor = args.config.model.get('subsampling_factor', 8)
    ds_seq_len = seq_len // subsample_factor
    args.config.model.attention_window_size = ds_seq_len // 2 # //2 because applied in both directions
    args.seq_len = args.__dict__.get('max_sequence_length', 3600000) # 10 hours

    include_per_recording_evaluations = args.__dict__.get('include_per_recording_evaluations', False)

    verbose = args.__dict__.get('verbose', True)   

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': args.config.get('model_class', args.model_class)}))
    model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)

    data = datasets_functions[args.dataset](args.split)

    # for idx, module in enumerate([el.attend.fn for el in model.layers]):
    #     module.return_attention_weights = True
    all_texts = []
    all_golds = []
    wer_data = []

    pbar = tqdm(range(len(data)), total=len(data)) #if verbose else range(len(data))
    for rec in pbar:
        if verbose: print(f'Processing {rec+1}/{len(data)}')
        
        if verbose: print('\n-------\n'+data[rec]['id']+'\n-------\n')

        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
        #args, model:SCConformerXL, spec:torch.Tensor, distracter_spec:torch.Tensor, distracter_spec_chunks_len:int, seq_len:int, window_len:int, buffer_len:int, tokenizer, use_tqdm=True
        
        #recordings = [i for i in range(len(data)) if i != rec]
        distracter_data = datasets_functions[args.distracter_dataset]('test')

        if args.within_recording:
            assert args.dataset == args.distracter_dataset, 'within_recording only makes sense when dataset and distracter_dataset are the same'

        all_logits = []
        for i in range(args.__dict__.get('repeats', 1)):
            if args.dataset == args.distracter_dataset and not args.within_recording:
                recordings = [i for i in range(len(distracter_data)) if i != rec]
            elif args.dataset == args.distracter_dataset and args.within_recording:
                recordings = [rec]
            else:
                recordings = [i for i in range(len(distracter_data))]

            # pick a random recording
            if args.distracter_dataset != 'no_context':
                distracter_rec_id = recordings[math.floor(torch.rand(1)*len(recordings))]
                distracter_spec, _ = distracter_data[distracter_rec_id]['process_fn'](distracter_data[distracter_rec_id])
            else:
                distracter_spec = torch.zeros_like(audio_spec)

            logits = shuffled_eval(
                args = args, 
                model = model, 
                spec = audio_spec,
                distracter_spec = distracter_spec, 
                distracter_spec_chunks_len = args.distracter_len,
                seq_len = args.seq_len,
                window_len = args.window_len,
                buffer_len = args.buffer_len,
                tokenizer = tokenizer,
                use_tqdm = True
            ) 
            all_logits.append(torch.as_tensor(logits))
        logits = torch.zeros_like(all_logits[0])
        for logit in all_logits:
            logits += logit.exp()
        logits = torch.log(logits / 3)
        out_text = decoder(logits)

        out = normalize(out_text).lower()
        
        if verbose: print(gold_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(gold_text)

        if include_per_recording_evaluations:
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
            wer_data.append({
                'recording': data[rec]['id'],
                'wer': wer,
                'words': words,
                'ins_rate': ins_rate,
                'del_rate': del_rate,
                'sub_rate': sub_rate
            })

        if args.__dict__.get('break_eval', False): break
        
        

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    if verbose: print(f'WER: {wer}')

    wer_data.append({
        'recording': 'all',
        'wer': wer,
        'words': words,
        'ins_rate': ins_rate,
        'del_rate': del_rate,
        'sub_rate': sub_rate
    })
    return wer_data, model_config
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repeats', type=int, default=1, help='repeat evaluation n times')
    parser.add_argument('--dataset', '-d', type=str, default='tedlium', choices=datasets_functions.keys())
    parser.add_argument('--distracter_dataset', '-dd', type=str, default='earnings22_full', choices=datasets_functions.keys())
    parser.add_argument('-wr', '--within_recording', action='store_true', help='sample distracter from within recording')

    parser.add_argument('-window_len', '--window_len', type=int, default=2048, help='window length')
    parser.add_argument('-buffer_len', '--buffer_len', type=int, default=2048, help='buffer length')
    parser.add_argument('-distracter_len', '--distracter_len', type=int, default=2048, help='distracter length')
    
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    parser.add_argument('-break', '--break_eval', action='store_true', help='break after first recording') 
    args = parser.parse_args()
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 

