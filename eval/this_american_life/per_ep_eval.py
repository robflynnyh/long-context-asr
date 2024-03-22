import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
import warnings
from omegaconf import OmegaConf
normalize = EnglishTextNormalizer()
import pandas as pd
from run import get_text_and_audio

def main(args):

    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config
    

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class(config=args.config, args=args))
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()



    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    data = get_text_and_audio('test') + get_text_and_audio('dev') + get_text_and_audio('train')

 
    for rec in tqdm(range(len(data)), total=len(data)):
        print(f'Processing {rec+1}/{len(data)}')
        print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
     
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)

        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)
        out = normalize(decoded[0]['text']).lower()
        
        print(gold_text, '\n', out, '\n\n')

        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
        
        cur_row = {
            'id': data[rec]['id'],
            'speakers': data[rec]['speakers'],
            'gold': gold_text,
            'hypothesis': out,
            'wer': wer,
            'words': words,
            'ins_rate': ins_rate,
            'del_rate': del_rate,
            'sub_rate': sub_rate,
            'checkpoint_path': args.checkpoint,
            'seq_len': args.seq_len,
            'overlap': args.overlap,
            'model_class': args.model_class
        }
        df = pd.DataFrame([cur_row])
        pd_exists = os.path.exists(args.save)
        df.to_csv(args.save, mode='a', header=not pd_exists) if args.save != '' else None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    parser.add_argument('-save', '--save', type=str, default='./output.csv', help='path to save output')

    args = parser.parse_args()
    main(args)
    
