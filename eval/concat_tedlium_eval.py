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
import random
from einops import rearrange

from earnings22_full.run import get_text_and_audio as get_text_and_audio_earnings22_full
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16
from this_american_life.run import get_text_and_audio as get_text_and_audio_this_american_life
from spotify.run import get_text_and_audio as get_text_and_audio_spotify

datasets_functions = {
    'tedlium': get_text_and_audio_tedlium,
}

@torch.inference_mode()
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
    concat_from = args.__dict__.get('concat_from', 'middle')
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)

    data = datasets_functions[args.dataset](args.split)

    # for idx, module in enumerate([el.attend.fn for el in model.layers]):
    #     module.return_attention_weights = True
    all_texts = []
    all_golds = []
    wer_data = []

    all_specs = [data[i]['process_fn'](data[i])[0] for i in range(len(data))]

    pbar = tqdm(range(len(data)), total=len(data)) #if verbose else range(len(data))
    for rec in pbar:
        if verbose: print(f'Processing {rec+1}/{len(data)}')
        
        if verbose: print('\n-------\n'+data[rec]['id']+'\n-------\n')

        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
        #args, model:SCConformerXL, spec:torch.Tensor, distracter_spec:torch.Tensor, distracter_spec_chunks_len:int, seq_len:int, window_len:int, buffer_len:int, tokenizer, use_tqdm=True
        
        #recordings = [i for i in range(len(data)) if i != rec]
        
        

        all_logits = []
        for i in range(args.__dict__.get('repeats', 1)):
            recordings = [i for i in range(len(data)) if i != rec]
            random.shuffle(recordings)
            all_specs_cur = [all_specs[i] for i in recordings]
            if concat_from == 'middle':
                left = all_specs_cur[:len(all_specs_cur)//2]
                right = all_specs_cur[len(all_specs_cur)//2:]
            elif concat_from == 'left':
                left = all_specs_cur
                right = []
            elif concat_from == 'right':
                left = []
                right = all_specs_cur
            specs = left + [audio_spec] + right
            specs = torch.cat(specs, dim=-1) 
          
            logits = model(specs.to(device))['final_posteriors']
            downsampled_by = specs.shape[-1] / logits.shape[-2]
            left_in_len = sum([el.shape[-1] for el in left])
            ds_left_len = int(left_in_len / downsampled_by)
            cur_spec_in_len = audio_spec.shape[-1]
            ds_cur_spec_len = int(cur_spec_in_len / downsampled_by)
            logits = logits[:, ds_left_len:ds_left_len+ds_cur_spec_len, :].clone()
            all_logits.append(torch.as_tensor(logits))

        logits = torch.zeros_like(all_logits[0])
        for logit in all_logits:
            logits += logit.exp()
        logits = torch.log(logits / args.__dict__.get('repeats', 1))
   
        out_text = decoder(rearrange(logits, '() n v -> n v'))

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
    parser.add_argument('--concat_from', '-cf', type=str, default='middle', choices=['middle', 'left', 'right'], help='concat from middle, left or right')

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

