import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model, get_model_class
from lcasr.utils.helpers import load_text
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
import warnings
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.utils.omegaconf import OmegaConf
normalize = EnglishTextNormalizer()
audio_EXT = '.wav'

paths_dir = os.path.join(os.path.dirname(__file__), '../paths.yaml')
if os.path.exists(paths_dir):
    paths = OmegaConf.load(paths_dir)
    ROOT_PATH = paths.floras50.root
else:
    warnings.warn('paths.yaml not found, using default path for floras50 dataset')
    ROOT_PATH = '/mnt/parscratch/users/acp21rjf/floras50_eval/'
   


def fetch_data(audio_path:str, txt_path:str):
    audio_files = [{
        'meeting': el.replace(audio_EXT, ''),
        'path': os.path.join(audio_path, el)
        } for el in os.listdir(audio_path) if el.endswith(audio_EXT)]

    text_files = [{
        'meeting': el['meeting'],
        'text': load_text(os.path.join(txt_path, el['meeting'] + '.txt'))
        } for el in audio_files]
 
    return audio_files, text_files



def preprocess_transcript(text:str):
    text = text.replace('…', '')
    text = text.replace(',', '')
    text = text.replace('-', ' ')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = re.sub(' +', ' ', text)
    return normalize(text).lower()


def process_text_and_audio_fn(rec_dict): return processing_chain(rec_dict['audio']), preprocess_transcript(rec_dict['text'])

def get_text_and_audio(split):
    assert split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    data_path = os.path.join(ROOT_PATH, split)
    audio_path = os.path.join(data_path, 'audio')
    text_path = os.path.join(data_path, 'text')
    audio_files, text_files = fetch_data(audio_path=audio_path, txt_path=text_path)
    return_data = []
    for rec in range(len(audio_files)):
        return_data.append({
            'id': audio_files[rec]['meeting'],
            'text': text_files[rec]['text'], 
            'audio': audio_files[rec]['path'], 
            "process_fn": process_text_and_audio_fn
        })
    return_data = sorted(return_data, key=lambda x: x['id'])
    return return_data


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

    # if args.pad_to != 0 and hasattr(model, 'use_padded_forward'):
    #     model.use_padded_forward(pad_to = args.pad_to)

 
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)

    data = get_text_and_audio(args.split)
   
    
    all_texts = []
    all_golds = []
    for rec in tqdm(range(len(data)), total=len(data)):
        print(f'Processing {rec+1}/{len(data)}')
        print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
     
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
        
        out_text = decoder(torch.as_tensor(logits))
        out = normalize(out_text).lower()
        
        
        print(gold_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(gold_text)
        

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')


    return wer, model_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='/mnt/parscratch/users/acp21rjf/lcasr-10s/step_105360_repeat_1.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    args = parser.parse_args()
    main(args)
    
