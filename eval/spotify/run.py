import lcasr
import torch, numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.audio_tools import processing_chain, total_seconds, total_frames

from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder

import torchaudio

from einops import rearrange
import os
import wandb
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



def ctc_decoder(vocab, alpha, beta, arpa_path=''):
    arpa_path = None if arpa_path == '' else arpa_path
    alpha = None if arpa_path is None else alpha
    beta = None if arpa_path is None else beta
    decoder = build_ctcdecoder(vocab, kenlm_model_path=arpa_path, alpha=alpha, beta=beta)
    return decoder

def decode_beams_lm(
        logits_list, 
        decoder, 
        beam_width=100, 
        encoded_lengths=None,
        ds_factor=4
    ):
    decoded_data = []
    if encoded_lengths is None:
        encoded_lengths = [len(logits) for logits in logits_list]

    def proc_text_frame(tx_frame:Tuple[str, Tuple[int, int]]):
        text, frame = tx_frame
        return {'word': text, 'start': total_seconds(frame[0] * ds_factor), 'end': total_seconds(frame[1] * ds_factor)}

    for logits, length in zip(logits_list, encoded_lengths):
 
        beams = decoder.decode_beams(
            logits = logits[:length],
            beam_width = beam_width,
            # beam_prune_logp = beam_prune_logp, 
            # token_min_logp = token_min_logp,
            # prune_history = prune_history
        )
        decoded_data.append({
                'text': beams[0].text,
                'frames': [proc_text_frame(el) for el in beams[0].text_frames],
                'ngram_score': beams[0].lm_score - beams[0].logit_score,
                'am_score': beams[0].logit_score,
                'score': beams[0].lm_score # # score = ngram_score + am_score
        })

    return decoded_data, beams[0]

def fetch_data():
    all_data = load_pairs()
    data_items = all_data.items()
    audio_paths = []
    texts = []
    for i in range(4):
        sample = list(data_items)[i][1]
        audio_path = sample['audio']
        text = sample['txt']
        audio_paths.append(audio_path)
        texts.append(text)

    return audio_paths, texts


def load_audio(audio_file:str):
    spec = torch.load(audio_file).to(dtype=torch.float32)
    return spec

def load_text(text_path:str):
    jfile = load_json(text_path)
    txt = jfile['results'][-1]['alternatives'][0]['words']
    txt = ' '.join([el['word'] for el in txt]).lower().strip()
    return txt

@torch.no_grad()
def fetch_logits(args, model:SCConformerXL, spec:torch.Tensor, seq_len:int, overlap:int, tokenizer, use_tqdm=True):
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']
    seq_len = seq_len if seq_len < spec_n else spec_n
    overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']
    cache_len = args.cache_len if args.cache_len != -1 else args.config['training']['max_seq_len']
    #assert overlap == 0 or cache_len == 0, 'Cannot use overlap and cache_len at the same time'

    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'

    print(f'Using seq_len: {seq_len} and overlap: {overlap} and cache_len: {cache_len}')

    all_logits = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    
    logit_position = 0
    
    prev_cache = None
    last_ulen = None
    kill_next = False
    pbar = tqdm(range(0, spec_n, seq_len-overlap), total=len(range(0, spec_n, seq_len-overlap))) if use_tqdm else range(0, spec_n, seq_len-overlap)
    for i in pbar:
        audio_chunk = spec[:, :, i:i+seq_len]
        u_len = audio_chunk.shape[-1]

        #   print(u_len, last_ulen, kill_next)
        if kill_next:
            break
        if last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        # if u_len < (seq_len - overlap):
        #     continue

        audio_chunk = audio_chunk.to(model.device)
        out = model(
            audio_signal = audio_chunk,
            cached_kvs = prev_cache,
            cached_kv_lengths = None if prev_cache is None else torch.LongTensor([prev_cache.shape[1]] * prev_cache.shape[0]).to(prev_cache.device)
        )

        if cache_len != 0:
            prev_cache = out['kvs_to_cache'][:, -cache_len:].clone()

        logits = out['final_posteriors'].detach().cpu()
        # convert to prob
        logits = torch.exp(logits)
        ds_len = logits.shape[-2]

        ratio = u_len / ds_len
        overlap_ds = int(overlap / ratio)
        if i != 0:
            logit_position -= overlap_ds

        #logit_position = i 
        #print(logits.shape, logit_position, ds_len, all_logits.shape, all_logits[:,logit_position:].shape)
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 

        #print(logit_position, overlap_ds, '()', u_len, i, i+seq_len, audio_chunk.shape, ratio, ds_len)
        
        
    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    # convert to log 
    logits = torch.log(logits)
    # blank_id = logits.shape[-1]-1
    # logits = logits.argmax(dim=-1)[0].tolist()
    # print(tokenizer.decode([el for el in logits if el != blank_id]))
    
    return logits.squeeze(0).numpy()


def main(args):

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config
    

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    decoder = ctc_decoder(
        vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""],
        alpha = args.alpha,
        beta = args.beta,
        arpa_path = None,
    ) # "" = blank


    audio_files, text_files = fetch_data()
    
    all_texts = []
    all_golds = []

    for ix, (rec, tex) in tqdm(enumerate(zip(audio_files, text_files)), total=len(audio_files)):
        print(f'Processing {ix+1}/{len(audio_files)}') if args.verbose else None   

        audio_spec = load_audio(rec)
        gold_text = load_text(tex)
       
     
       
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
       
        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=args.beam_width, ds_factor=ds_factor)

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

    parser.add_argument('-log', '--log', type=str, default='')

    args = parser.parse_args()
    args.verbose = not args.not_verbose

    main(args)
    
