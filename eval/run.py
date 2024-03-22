import torch, argparse, lcasr
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
from tqdm import tqdm

from earnings22_full.run import get_text_and_audio as get_text_and_audio_earnings22_full
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16

datasets_functions = {
    'earnings22_full': get_text_and_audio_earnings22_full,
    'earnings22': get_text_and_audio_earnings22,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16
}


def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    for el in ['rev16', 'earnings22_full']: 
        if args.dataset == el: assert args.split == 'test', f'Split must be test for {el}'

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config

    if args.disable_flash_attention:    
        args.config.model.flash_attn = False

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': args.config.get('model_class', args.model_class)}))
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    data = datasets_functions[args.dataset](args.split)

    all_texts = []
    all_golds = []

    for rec in tqdm(range(len(data)), total=len(data)):
        print(f'Processing {rec+1}/{len(data)}')
        
        print('\n-------\n'+data[rec]['id']+'\n-------\n')
    
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
        
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
        
        ds_factor = audio_spec.shape[-1] / logits.shape[0]

        decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)
        out_text = decoded[0]['text']

        out = normalize(out_text).lower()
        
        print(gold_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(gold_text)
        

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')

    return wer, model_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())

    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    args = parser.parse_args()
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 

