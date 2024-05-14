import torch, argparse, lcasr
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.buffered_transcription import fetch_logits as buffered_eval
from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from lcasr.decoding.greedy import GreedyCTCDecoder
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
from tqdm import tqdm
import os
import pickle as pkl

from earnings22_full.run import get_text_and_audio as get_text_and_audio_earnings22_full
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16
from this_american_life.run import get_text_and_audio as get_text_and_audio_this_american_life

datasets_functions = {
    'earnings22_full': get_text_and_audio_earnings22_full,
    'earnings22': get_text_and_audio_earnings22,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16,
    'this_american_life': get_text_and_audio_this_american_life
}

@torch.no_grad()
def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config

    if args.__dict__.get('disable_flash_attention', False): args.config.model.flash_attn = False

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
        #if rec != 5: continue 
        if verbose: print(f'Processing {rec+1}/{len(data)}')
        
        if verbose: print('\n-------\n'+data[rec]['id']+'\n-------\n')

        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
        print(audio_spec.shape)
        # split into windows of size window_size
        window_size = args.window_size
        window_starts_and_ends = [(i, min(i+window_size, audio_spec.shape[-1])) for i in range(0, audio_spec.shape[-1], window_size)]
        # create nxn matrix for WERs
        wer_matrix = torch.zeros(len(window_starts_and_ends), len(window_starts_and_ends) + 1)

        logits = model(audio_spec.to(device))['final_posteriors']
        spec_n, logits_n = audio_spec.shape[-1], logits.shape[1]
        downsampled_by = spec_n / logits_n
        downsampled_window_starts_and_ends = [(int(start/downsampled_by), int(end/downsampled_by)) for start, end in window_starts_and_ends]
        print(downsampled_window_starts_and_ends)

        out_text = decoder(logits.squeeze(0))
        out = normalize(out_text).lower()

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
        print(wer*100)
        wer_matrix[:, -1] = wer*100
        unharmed_logits = logits

        target_transcript = gold_text
        unharmed_transcript = out
        trancript_matrix = []

        for i, (start, end) in enumerate(window_starts_and_ends):
            trancript_matrix.append([])
            for j, (start2, end2) in enumerate(window_starts_and_ends):
                mask_start, mask_end = start2, end2
                cur_audio_spec = audio_spec.clone()
                mask_val = audio_spec[:, :, mask_start:mask_end].mean().item()
                cur_audio_spec[:, :, mask_start:mask_end] = mask_val
                cur_logits = model(cur_audio_spec.to(device))['final_posteriors']
                cur_unharmed_logits = unharmed_logits.clone()
                downsampled_start, downsampled_end = int(start/downsampled_by), int(end/downsampled_by)
                cur_unharmed_logits[:, downsampled_start:downsampled_end] = cur_logits[:, downsampled_start:downsampled_end].clone()
                cur_logits = cur_unharmed_logits
                out_text = decoder(cur_logits.squeeze(0))
                out = normalize(out_text).lower()
                wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
                trancript_matrix[i].append(out)
                if i!=j: 
                    print(wer*100)
                    wer_matrix[i, j] = wer*100
                else: 
                    print(f'-- self-mask: -- {wer*100}')
                    wer_matrix[i, j] = wer*100

                

            print('---')

 

        if args.save != '':
                checkpoint_name = args.checkpoint.split('/')[-2]
                path_wer = os.path.join(args.save, f'{args.dataset}_{checkpoint_name}_{args.split}_recording_{rec}_wers.pt')
                path_transcripts = os.path.join(args.save, f'{args.dataset}_{checkpoint_name}_{args.split}_recording_{rec}_transcripts.pkl')
                torch.save(wer_matrix, path_wer)
                with open(path_transcripts, 'wb') as f:
                     pkl.dump({
                          'transcript_matrix': trancript_matrix,
                          'gold': gold_text,
                          'unharmed': unharmed_transcript,
                     }, f)
            

        if args.break_eval: break

  

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())

    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')
    parser.add_argument('-repeat', '--repeat', type=int, default=1, help='number of times to rerun evaluation')
    parser.add_argument('-window', '--window_size', type=int, default=2048, help='window size')

    parser.add_argument('-save', '--save', type=str, default='/mnt/parscratch/users/acp21rjf/spotify/context_attribution/')

    parser.add_argument('-break', '--break_eval', action='store_true', help='break after first recording') 
    args = parser.parse_args()
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 

