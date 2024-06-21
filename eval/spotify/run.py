import lcasr
import torch, numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.buffered_transcription import fetch_logits as buffered_eval
from lcasr.eval.utils import  decode_beams_lm

from lcasr.utils.audio_tools import processing_chain, total_seconds, total_frames
from lcasr.decoding.greedy import GreedyCTCDecoder
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


def fetch_data(items=24, seed=57):
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
        if data_items[i][1]['duration'] / 60 >= 20:
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


def preprocess_transcript(text:str):
    return normalize(text).lower()

def process_text_and_audio_fn(rec_dict): return rec_dict['audio'], preprocess_transcript(rec_dict['text'])

def get_text_and_audio(split, **kwargs):
    audio_files, text_files = fetch_data()
  
    return_data = []
    for rec in range(len(audio_files)):
        return_data.append({
            'text': load_text(text_files[rec]), 
            'audio': load_audio(audio_files[rec]), 
            "process_fn": process_text_and_audio_fn,
            "id": text_files[rec],
        })
    return return_data

def main(args):

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config
    
    eval_fn = moving_average_eval
    if args.__dict__.get('evaluation_mode', 'averaged_moving_window') == 'windowed_attention':
        seq_len = args.seq_len
        subsample_factor = args.config.model.get('subsampling_factor', 8)
        ds_seq_len = seq_len // subsample_factor
        args.config.model.attention_window_size = ds_seq_len // 2 # //2 because applied in both directions
        args.seq_len = args.__dict__.get('max_sequence_length', 3600000) # 10 hours
    if args.__dict__.get('evaluation_mode', 'averaged_moving_window') == 'buffered': eval_fn = buffered_eval

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': args.config.get('model_class', args.model_class)}))

    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)


    audio_files, text_files = fetch_data()
    
    all_texts = []
    all_golds = []

    for ix, (rec, tex) in tqdm(enumerate(zip(audio_files, text_files)), total=len(audio_files)):
        print(f'Processing {ix+1}/{len(audio_files)}') if args.verbose else None   

        audio_spec = load_audio(rec)
        gold_text = load_text(tex)
        CTC_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
        meta_loss_fn = lambda a, b, d: (torch.nn.functional.cross_entropy(b, a, reduction='mean'))

        #with torch.set_grad_enabled(True):
        for repeat in range(args.__dict__.get('repeat', 1)):
            audio_spec = audio_spec.to(device)
            audio_spec.requires_grad = True
            logits = model(audio_spec)['final_posteriors']
            #   print(logits)
            probs = torch.as_tensor(logits)
            
            tokens = torch.as_tensor(tokenizer.encode(gold_text))[None]
            
            cosine_loss_fn = torch.nn.CosineSimilarity(dim=-1)
            
            loss = CTC_loss_fn(
                log_probs=probs.transpose(0,1), 
                targets=tokens, 
                input_lengths = torch.LongTensor([probs.shape[1]]),
                target_lengths = torch.LongTensor([tokens.shape[1]])
            ) 
            #print(loss)
            update = torch.autograd.grad(loss, inputs = model.reprs, retain_graph=True)[0]
            #print(update.shape)
            q, vq_indices, _ = model.grad_vq(update)
            layer1_params = [p for p in model.layers[0].parameters()]
            weight_grads = torch.autograd.grad(outputs=model.reprs, inputs=layer1_params, grad_outputs=q)
            for p, g in zip(layer1_params, weight_grads):
                p.data = p.data - g * 1e-3
            
            vals, indices = model.grad_pred.softmax(dim=-1).max(dim=-1)
            print(vq_indices, indices)
            #print(vq_indices.shape, model.grad_pred.shape)
            ce_loss = meta_loss_fn(vq_indices.squeeze(0), model.grad_pred.squeeze(0), None)
            print(ce_loss, 'ce_loss')
    
            # for p, u in zip(d_inputs, update):
                #     p.data.add_(-1e-2 * u)
       
        out_text = decoder(logits.squeeze(0))

        all_text = normalize(out_text).lower()
        gold_text = normalize(gold_text).lower()    
        print(gold_text) if args.verbose else None
        print(all_text) if args.verbose else None
        all_texts.append(all_text)
        all_golds.append(gold_text)

        if args.__dict__.get('break_eval', False):
            break
 

        
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


    parser.add_argument('-repeat', '--repeat', type=int, default=1, help='number of times to rerun evaluation')
    parser.add_argument('-break', '--break_eval', action='store_true', help='break after each evaluation')
    parser.add_argument('-eval_mode', '--evaluation_mode', type=str, default='averaged_moving_window', choices=['averaged_moving_window', 'windowed_attention', 'buffered'])


    parser.add_argument('-log', '--log', type=str, default='')

    args = parser.parse_args()
    args.verbose = not args.not_verbose

    main(args)
    
