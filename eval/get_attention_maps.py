import torch, argparse, lcasr
from lcasr.utils.general import load_model, get_model_class
from tqdm import tqdm
from lcasr.components.attention import CollectFlashAttentionProbs

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


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config

    if args.__dict__.get('disable_flash_attention', False): args.config.model.flash_attn = False

    seq_len = args.seq_len
        # subsample_factor = args.config.model.get('subsampling_factor', 8)
        # ds_seq_len = seq_len // subsample_factor
        # args.config.model.attention_window_size = ds_seq_len // 2 # //2 because applied in both directions
        # args.seq_len = args.__dict__.get('max_sequence_length', 3600000) # 10 hours


    verbose = args.__dict__.get('verbose', True)   

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': args.config.get('model_class', args.model_class)}))
    model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')

    assert torch.cuda.is_available(), 'currently this script is written for using flash attention on GPU only 0-:'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    flash_attn_modules = [layer.attend.fn.flash_attn_fn for layer in model.layers]
    AttentionCollector = CollectFlashAttentionProbs(attn_modules = flash_attn_modules)


    data = datasets_functions[args.dataset](args.split)

    all_texts = []
    all_golds = []
    wer_data = []

    pbar = tqdm(range(len(data)), total=len(data)) #if verbose else range(len(data))
    for rec in pbar:
        if verbose: print(f'Processing {rec+1}/{len(data)}')
        
        if verbose: print('\n-------\n'+data[rec]['id']+'\n-------\n')
    
        audio_spec, _ = data[rec]['process_fn'](data[rec])
        model(audio_spec.to(device))
        attention_maps = AttentionCollector()
        print(len(attention_maps))
        #print(attention_maps[0].shape, attention_maps[1].shape, len(attention_maps))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())

    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    args = parser.parse_args()
    assert args.model_class == 'SCConformerXL', 'Currently only SCConformerXL is supported for attention map extraction'
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 

