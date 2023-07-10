import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.audio_tools import SimpleDataloader, processing_chain, total_seconds
from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder

import torchaudio

from einops import rearrange
import os
import wandb
import re

from wer import word_error_rate_detail 

TEST_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/test/'


def remove_punctuation(text): # replace with space
    # punctuation = everything except "'" which is used in contractions
    text = re.sub(r"[^\w\d'\s]+",' ',text)
    # remove multiple spaces
    text = re.sub(' +', ' ', text)
    return text

def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def proc_stm_and_timings(stm_path:str):
    stm = open_stm(stm_path)
    all_text = ""
    timings = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            continue
        all_text += text + ' '
        timings.append({'start': float(start), 'end': float(end)})
    all_text = all_text.strip()
    # regex to do all of the above
    # i.e replace space followed by a apostrophe followed by a letter with just the apostrophe and letter
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    # remove multiple spaces
    all_text = re.sub(r" +", r" ", all_text)
    return all_text, timings

def ctc_decoder(vocab):  
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)
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

def fetch_test_data(path:str = TEST_PATH):
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



def fetch_logits(args, model:SCConformerXL, spec:torch.Tensor, seq_len:int, overlap:int, tokenizer):
    spec_n = spec.shape[-1]
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']
    seq_len = seq_len if seq_len < spec_n else spec_n
    overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']

    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits = torch.zeros((1, spec_n//4 + 10, tokenizer.vocab_size() + 1))
    logit_count = torch.zeros((1, spec_n//4 + 10, tokenizer.vocab_size() + 1))
    
    logit_position = 0

    logit_list = []
    for i in range(0, spec_n, seq_len-overlap):
        audio_chunk = spec[:, :, i:i+seq_len]
        u_len = audio_chunk.shape[-1]
        audio_chunk = audio_chunk.to(model.device)
        out = model(audio_chunk)
        logits = out['final_posteriors'].detach().cpu()
        # convert to prob
        logits = torch.exp(logits)
        ds_len = logits.shape[-2]

        ratio = u_len / ds_len
        overlap_ds = int(overlap / ratio)
        if i != 0:
            logit_position -= overlap_ds

        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 
        
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


def parse_utterances(decoded_frames:List[Dict[str, any]], timings:List[Dict[str, int]]):    
    parsed_decoded_frames = []
    buffer = 1.0
    for frame in decoded_frames:
        start, end = frame['start'], frame['end']
        for timing in timings:
            t_start, t_end = timing['start'] - buffer, timing['end'] + buffer
            if start >= t_start and end <= t_end:
                parsed_decoded_frames.append(frame)
                break
    all_text = remove_punctuation(" ".join(el['word'] for el in parsed_decoded_frames))
    return all_text, parsed_decoded_frames            


def main(args):
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config
    

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    

    decoder = ctc_decoder([tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]) # "" = blank


    audio_files, text_files = fetch_test_data()
    paired = dict(zip(audio_files, text_files))
    
    all_texts = []
    all_golds = []
    for rec in range(len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}')

        audio_spec = load_audio(audio_files[rec])
        print('\n\n'+paired[audio_files[rec]]+'\n\n')
        
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
      
        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=args.beam_width, ds_factor=ds_factor)
        
        stm_path = paired[audio_files[rec]]
        gold_text, timings = proc_stm_and_timings(stm_path=stm_path)
        all_text, frames = parse_utterances(decoded_frames = decoded[0]['frames'], timings = timings)
        print(all_text)
        all_texts.append(all_text)
        all_golds.append(gold_text)
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')

    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-beams', '--beam_width', type=int, default=10, help='beam width for decoding')

    args = parser.parse_args()
    main(args)
    