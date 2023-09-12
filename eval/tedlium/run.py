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

from wer import word_error_rate_detail 
from postprocess import post_process
import re

TEST_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/test/'
DEV_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/dev/'


from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def proc_stm_and_timings(stm_path:str):
    stm = open_stm(stm_path)
    all_text = ""
    timings = []
    remove_timings = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            remove_timings.append({'start': float(start), 'end': float(end)})
            continue
        all_text += text + ' '
        timings.append({'start': float(start), 'end': float(end)})
    all_text = all_text.strip()
    # regex to do all of the above
    # i.e replace space followed by a apostrophe followed by a letter with just the apostrophe and letter
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    # remove multiple spaces
    all_text = re.sub(r" +", r" ", all_text)
    return all_text, timings, remove_timings

def fetch_utterances(stm_path:str, spectogram:torch.Tensor):
    stm = open_stm(stm_path)
    utterances = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            continue
        utterances.append({
            'start': float(start), 
            'end': float(end), 
            'text': text, 
            'start_frame': total_frames(float(start)), 
            'end_frame': total_frames(float(end)),
            'spectogram': spectogram[:, :, total_frames(float(start)):total_frames(float(end))]
        })
    
    all_text = " ".join([el['text'] for el in utterances])
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    all_text = re.sub(r" +", r" ", all_text)
    
    
    return utterances, all_text

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

def fetch_data(path:str = TEST_PATH):
    audio_path = os.path.join(path, 'sph')
    audio_files = [os.path.join(audio_path, el) for el in os.listdir(audio_path) if el.endswith('.sph')]
    audio_files.sort()
    text_path = os.path.join(path, 'stm')
    text_files = [os.path.join(text_path, el) for el in os.listdir(text_path) if el.endswith('.stm')]
    text_files.sort()
    assert len(audio_files) == len(text_files), 'Number of audio files and text files must match'
    return audio_files, text_files


def load_audio(audio_file:str):
    spec = processing_chain(audio_file)
    return spec



@torch.no_grad()
def fetch_logits(args, model:SCConformerXL, spec:torch.Tensor, seq_len:int, overlap:int, tokenizer, use_tqdm=True):
    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']
 
    if seq_len > spec_n:
        seq_len = spec_n
        overlap = 0
    else:
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


def parse_utterances_(decoded_frames:List[Dict[str, any]], timings:List[Dict[str, int]]):    
    parsed_decoded_frames = []
    buffer = 0.0
    for frame in decoded_frames:
        start, end = frame['start'], frame['end']
        to_keep = True
        for timing in timings:
            t_start, t_end = timing['start'] - buffer, timing['end'] + buffer
            if start >= t_start and end <= t_end:
                to_keep = False
                break
        if to_keep:
            parsed_decoded_frames.append(frame)
    #print(" ".join(el['word'] for el in parsed_decoded_frames))
    all_text = post_process(" ".join(el['word'] for el in parsed_decoded_frames))
    return all_text, parsed_decoded_frames            

def parse_utterances(decoded_frames:List[Dict[str, any]], timings:List[Dict[str, int]]):    
    parsed_decoded_frames = []
    buffer = 10.0
    for frame in decoded_frames:
        start, end = frame['start'], frame['end']
        for timing in timings:
            t_start, t_end = timing['start'] - buffer, timing['end'] + buffer
            if start >= t_start and end <= t_end:
                parsed_decoded_frames.append(frame)
                break
    #print(" ".join(el['word'] for el in parsed_decoded_frames))
    all_text = post_process(" ".join(el['word'] for el in parsed_decoded_frames))
    return all_text, parsed_decoded_frames     

def zero_out_spectogram(spec:torch.Tensor, remove_timings:List[Dict[str, int]]):
    buffer = -0.5

    for timing in remove_timings:
        start, end = timing['start'] - buffer, timing['end'] + buffer

        start_frame, end_frame = map(total_frames, [start, end])
        spec[:,:,start_frame:end_frame] = 0

    return spec


def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    data_path = TEST_PATH if args.split == 'test' else DEV_PATH

    
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

    arpa_path = args.arpa_path if args.beam_width != 1 else ''
    decoder = ctc_decoder(
        vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""],
        alpha = args.alpha,
        beta = args.beta,
        arpa_path = args.arpa_path,
    ) # "" = blank


    audio_files, text_files = fetch_data(path=data_path)
    paired = dict(zip(audio_files, text_files))
    
    all_texts = []
    all_golds = []

    if not args.single_utterance:
        for rec in tqdm(range(len(audio_files)), total=len(audio_files)):
            print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None   

            audio_spec = load_audio(audio_files[rec])
            print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
            stm_path = paired[audio_files[rec]]
            gold_text, timings, remove_timings = proc_stm_and_timings(stm_path=stm_path)

            audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings)
            import time
            stime = time.time()
            logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
            etime = time.time()
            print(f'Inference time: {etime-stime}')
            ds_factor = audio_spec.shape[-1] / logits.shape[0]
            decoded, bo = decode_beams_lm([logits], decoder, beam_width=args.beam_width, ds_factor=ds_factor)

            all_text = normalize(decoded[0]['text']).lower()
            gold_text = normalize(gold_text).lower()    
            print(gold_text) if args.verbose else None
            print(all_text) if args.verbose else None
            all_texts.append(all_text)
            all_golds.append(gold_text)
            #break
    else:
        for rec in tqdm(range(len(audio_files)), total=len(audio_files)):

            print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None

            audio_spec = load_audio(audio_files[rec])
            print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
            stm_path = paired[audio_files[rec]]
            utterances, gold_text = fetch_utterances(stm_path=stm_path, spectogram=audio_spec)
           
            out_texts = []
            for utterance in tqdm(utterances):
                logit = fetch_logits(args, model, utterance['spectogram'], utterance['spectogram'].shape[-1], 0, tokenizer, use_tqdm=False)
                ds_factor = utterance['spectogram'].shape[-1] / logit.shape[0]
                decoded, bo = decode_beams_lm([logit], decoder, beam_width=args.beam_width, ds_factor=ds_factor)
                out_text = normalize(decoded[0]['text']).lower().strip()
                out_texts.append(out_text)
            all_text = " ".join(out_texts).strip()#
            gold_text = normalize(gold_text).lower().strip()
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
    
