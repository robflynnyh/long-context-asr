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




def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines


def log(args, wer, beam_width, alpha, beta, cur_prune_less_than_val, top_am):
    log_path = args.log_path
    with open(log_path, 'a') as f:
        f.write(f'{args.checkpoint} - wer:{wer} b:{beam_width} a:{alpha} b:{beta} p:{cur_prune_less_than_val} amt:{top_am}\n')

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

  
    alpha_range = [0.05, 0.3] # between 
    beta_range = [0.4, 1.1] # between
    beams = [25, 50] # either 
    prune_less_than_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # either
    top_am_thresh = [-4, -5, -6. -7, -8, -9, -10, -11, -12] # either

    while True:
        cur_alpha = np.random.uniform(alpha_range[0], alpha_range[1])
        cur_beta = np.random.uniform(beta_range[0], beta_range[1])
        cur_beam = np.random.choice(beams)
        cur_prune_less_than_val = np.random.choice(prune_less_than_vals)
        cur_top_am_thresh = np.random.choice(top_am_thresh)

        hyps = []
        golds = []
        for i in range(1):
            logits = all_logits[i]['logits']
            gold_text = all_logits[i]['gold']

            bs = beam_search.BeamSearch(
                tokenizer = tokenizer,
                beam_width = cur_beam,
                log_probs = logits,
                language_model = language_model,
                blank_id = tokenizer.vocab_size(),
                alpha = cur_alpha,
                beta = cur_beta,
                debug = False,
                prune_less_than_val = cur_prune_less_than_val,
                top_am_threshold = cur_top_am_thresh,
            )
            bs.run_search(use_tqdm=True)
            text_out = post_process(text = bs.return_text(idx = 0))
            hyps.append(text_out)
            golds.append(gold_text)
           
        
        wer = word_error_rate_detail(hypotheses=hyps, references=golds)[0]
        print(wer, cur_beam, cur_alpha, cur_beta)
        log(
            args = args,
            wer = wer,
            beam_width = cur_beam,
            alpha = cur_alpha,
            beta = cur_beta,
            cur_prune_less_than_val = cur_prune_less_than_val,
            top_am = cur_top_am_thresh,
        )
        #print('0.08997, 0.05626')

        
    #wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    #print(f'WER: {wer}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='/mnt/parscratch/users/acp21rjf/language_modelling/spotipile/5e4/step_108000.pt', help='path to checkpoint')
    
    parser.add_argument('-beams', '--beam_width', type=int, default=1, help='beam width for decoding')
    parser.add_argument('-logits', '--logits_path', type=str, default='./logits/logits.pkl', help='path to logits')
    parser.add_argument('-log', '--log_path', type=str, default='tlm_beam.log', help='path to log file')


    # wer:0.06839218375042852 b:10 a:0.25131218509148656 b:0.8917145455147044
    # wer:0.06804936578676722 b:5 a:0.07152317088066645 b:0.9052812115538087

    args = parser.parse_args()
    main(args)
    
#0.10381063195403971