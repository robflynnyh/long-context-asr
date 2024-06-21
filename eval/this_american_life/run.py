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

paths_dir = os.path.join(os.path.dirname(__file__), '../paths.yaml')
if os.path.exists(paths_dir):
    paths = OmegaConf.load(paths_dir)
    AUDIO_PATH = paths.this_american_life.audio
    TRAIN_PATH = paths.this_american_life.train
    DEV_PATH = paths.this_american_life.dev
    TEST_PATH = paths.this_american_life.test
else:
    warnings.warn('paths.yaml not found, using default paths for this american life dataset')
    AUDIO_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/audio'
    TRAIN_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/train-transcripts-aligned.json'
    DEV_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/valid-transcripts-aligned.json'
    TEST_PATH = '/mnt/parscratch/users/acp21rjf/this_american_life/test-transcripts-aligned.json'

EXT = '.mp3'


def fetch_data(txt_path:str):
    with open(txt_path, 'r') as f:
        txt_json = json.load(f)

    episodes = list(txt_json.keys())
    audio_files = [{'path':os.path.join(AUDIO_PATH, el.split('-')[-1] + EXT), 'id': el} for el in episodes]
    text = [{'id': el, 'text': " ".join([el2['utterance'] for el2 in txt_json[el]])} for el in episodes]
    speakers = [len(set([el2['speaker'] for el2 in txt_json[el]])) for el in episodes]

    return audio_files, text, speakers


def preprocess_transcript(text:str): return normalize(text).lower()


def process_text_and_audio_fn(rec_dict): return processing_chain(rec_dict['audio']), preprocess_transcript(rec_dict['text'])

def get_text_and_audio(split, **kwargs):
    if split == 'train':
        data_path = TRAIN_PATH
    elif split == 'dev':
        data_path = DEV_PATH
    elif split == 'test':
        data_path = TEST_PATH
    elif split == 'all':
        return get_text_and_audio('train') + get_text_and_audio('dev') + get_text_and_audio('test')
    else:
        raise ValueError(f'Invalid split: {split}')
     
    audio_files, text, speakers = fetch_data(txt_path=data_path)
    return_data = []
    for rec in range(len(audio_files)):
        assert audio_files[rec]['id'] == text[rec]['id'], f'Episode names do not match: {audio_files[rec]["id"]}, {text[rec]["id"]}'
        return_data.append({
            'id': audio_files[rec]['id'],
            'text': text[rec]['text'], 
            'audio': audio_files[rec]['path'], 
            "process_fn": process_text_and_audio_fn,
            'speakers': speakers[rec]
        })
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

    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    data = get_text_and_audio(args.split)
    
    all_texts = []
    all_golds = []
    for rec in tqdm(range(len(data)), total=len(data)):
        print(f'Processing {rec+1}/{len(data)}')
        print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
     
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)

        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)
        out = normalize(decoded[0]['text']).lower()
        
        print(gold_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(gold_text)
        

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')


    return wer, model_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    args = parser.parse_args()
    main(args)
    
