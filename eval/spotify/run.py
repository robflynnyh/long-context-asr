import lcasr
import torch, numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
from lcasr.eval.utils import fetch_logits, decode_beams_lm

from lcasr.utils.audio_tools import processing_chain, total_seconds, total_frames

from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder

import torchaudio

from einops import rearrange
import os
import wandb
import random
import re

from lcasr.eval.wer import word_error_rate_detail 
import re

from lcasr.utils.audio_tools import load_pairs
from lcasr.utils.audio_tools import load_json


from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines


def fetch_data(items=24, seed=41):
    all_data = load_pairs()
    data_items = all_data.items()
    audio_paths = []
    texts = []
    # randomise order with fixed seed
    random.seed(seed)
    data_items = list(data_items)
    random.shuffle(data_items)
    i=0
    while len(audio_paths) < items:
        if data_items[i][1]['duration'] / 60 >= 60:
            sample = data_items[i][1]
            audio_path = sample['audio']
            text = sample['txt']
            audio_paths.append(audio_path)
            texts.append(text)
        i+=1

    return audio_paths, texts


def load_audio(audio_file:str):
    spec = torch.load(audio_file).to(dtype=torch.float32)
    return spec

def load_text(text_path:str):
    jfile = load_json(text_path)
    txt = jfile['results'][-1]['alternatives'][0]['words']
    txt = ' '.join([el['word'] for el in txt]).lower().strip()
    return txt


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


    audio_files, text_files = fetch_data()
    
    all_texts = []
    all_golds = []

    for ix, (rec, tex) in tqdm(enumerate(zip(audio_files, text_files)), total=len(audio_files)):
        print(f'Processing {ix+1}/{len(audio_files)}') if args.verbose else None   

        audio_spec = load_audio(rec)
        gold_text = load_text(tex)
       
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
       
        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)

        all_text = normalize(decoded[0]['text']).lower()
        gold_text = normalize(gold_text).lower()    
        print(gold_text) if args.verbose else None
        print(all_text) if args.verbose else None
        all_texts.append(all_text)
        all_golds.append(gold_text)
 

        
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')

    if args.log != '':
        with open(args.log, 'a') as f:
            f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')

    return wer, model_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-beams', '--beam_width', type=int, default=1, help='beam width for decoding')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')

    parser.add_argument('-arpa', '--arpa_path', type=str, default='', help='path to arpa file')
    parser.add_argument('-alpha', '--alpha', type=float, default=0.5, help='alpha for lm')
    parser.add_argument('-beta', '--beta', type=float, default=0.8, help='beta for lm')

    parser.add_argument('-single_utt', '--single_utterance', action='store_true', help='single utterance decoding')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    parser.add_argument('-log', '--log', type=str, default='')

    args = parser.parse_args()
    args.verbose = not args.not_verbose

    main(args)
    
