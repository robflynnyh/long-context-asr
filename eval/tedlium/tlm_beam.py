import lcasr
import torch, numpy as np
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
from lming.decoding import beam_search

from lcasr.utils.audio_tools import processing_chain, total_seconds

from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder

import torchaudio
import lming
from lming.utils import general

from einops import rearrange
import os
import wandb
import re

from wer import word_error_rate_detail 
from postprocess import post_process
import re
import pickle as pkl

from postprocess import post_process

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


def log(args, wer, beam_width, alpha, beta):
    log_path = args.log_path
    with open(log_path, 'a') as f:
        f.write(f'{args.checkpoint} - wer:{wer} b:{beam_width} a:{alpha} b:{beta}\n')

def main(args):

    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    checkpoint['model'] = general.convert_from_ddp(checkpoint['model'])
    model_config = checkpoint['config']
    args.config = model_config
    
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()

    model = general.load_model(config=model_config, vocab_size=tokenizer.vocab_size())
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()


    language_model = beam_search.LanguageModel(
        model = model,
        bos_id = tokenizer.bos_id(),
        device = device,
    )
    
    with open(args.logits_path, 'rb') as f:
        all_logits = pkl.load(f)

    
    gold_text = all_logits[0]['gold']
    #print(gold_text)
    hyps = []
    golds = []

    for i in range(len(all_logits)):
        print(f'logits {i+1}/{len(all_logits)}')
        logits = all_logits[i]['logits']
        gold_text = all_logits[i]['gold']
        alpha = 0.25
        beta = 0.65
        p = 2.0
        beam_w = 25
        top_am_threshold = -6
        # wer:0.06804936578676722 b:5 a:0.07152317088066645 b:0.9052812115538087
        # wer:0.06839218375042852 b:10 a:0.25131218509148656 b:0.8917145455147044
        # wer:0.07147754542338018 b:25 a:0.15827085988604167 b:0.8139027466641017
        #--
        # wer:0.04832585433206766 b:25 a:0.23025491771314216 b:0.8262159074639039 p:1.5

        bs = beam_search.BeamSearch(
            tokenizer = tokenizer,
            beam_width = beam_w,
            log_probs = logits,
            language_model = language_model,
            blank_id = tokenizer.vocab_size(),
            alpha = alpha,
            beta = beta,
            debug=False,
            prune_less_than_val = p,
            top_am_threshold = top_am_threshold,
        )
        bs.run_search(use_tqdm=True)
        #text_out = post_process(text = bs.return_text(idx = 0))
        text_out = normalize(bs.return_text(idx = 0)).lower()
        gold_text = normalize(gold_text).lower()
        print(f'out: {text_out}')
        hyps.append(text_out)
        golds.append(gold_text)
        #break
        
        

    wer = word_error_rate_detail(hypotheses=hyps, references=golds)[0]
    print(f'WER: {wer}')

    # alpha_range = [0.0, 0.5] # between 
    # beta_range = [0.6, 1.2] # between
    # beams = [10, 5, 15] # either 10 or 25

    # while True:
    #     cur_alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    #     cur_beta = np.random.uniform(beta_range[0], beta_range[1])
    #     cur_beam = np.random.choice(beams)

    #     hyps = []
    #     golds = []
    #     for i in range(3):
    #         logits = all_logits[i]['logits']
    #         gold_text = all_logits[i]['gold']

    #         bs = beam_search.BeamSearch(
    #             tokenizer = tokenizer,
    #             beam_width = cur_beam,
    #             log_probs = logits,
    #             language_model = language_model,
    #             blank_id = tokenizer.vocab_size(),
    #             alpha = cur_alpha,
    #             beta = cur_beta,
    #             debug=False
    #         )
    #         bs.run_search(use_tqdm=True)
    #         text_out = post_process(text = bs.return_text(idx = 0))
    #         hyps.append(text_out)
    #         golds.append(gold_text)
           
        
    #     wer = word_error_rate_detail(hypotheses=hyps, references=golds)[0]
    #     print(wer, cur_beam, cur_alpha, cur_beta)
    #     log(
    #         args = args,
    #         wer = wer,
    #         beam_width = cur_beam,
    #         alpha = cur_alpha,
    #         beta = cur_beta
    #     )
        #print('0.08997, 0.05626')

        
    #wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    #print(f'WER: {wer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='/mnt/parscratch/users/acp21rjf/language_modelling/spotipile/4p5e4_512_512c/step_684000.pt', help='path to checkpoint')
    
    parser.add_argument('-beams', '--beam_width', type=int, default=1, help='beam width for decoding')
    parser.add_argument('-logits', '--logits_path', type=str, default='./logits/test_logits.pkl', help='path to logits')
    parser.add_argument('-log', '--log_path', type=str, default='tlm_beam.log', help='path to log file')


    # wer:0.06839218375042852 b:10 a:0.25131218509148656 b:0.8917145455147044
    # wer:0.06804936578676722 b:5 a:0.07152317088066645 b:0.9052812115538087

    args = parser.parse_args()
    main(args)
    
#0.10381063195403971