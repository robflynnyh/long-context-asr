import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
import traceback
from lcasr.utils.dataloading import VariableBatchSimpleDataloader, chunk_spectogram, chunk_text_json, reset_seen_ids
from lcasr.utils.hooks import add_debug_backwards_hooks
from lcasr.utils.scheduling import CosineLRScheduler, SequenceWarmupManager
from lcasr.utils.helpers import exists
from lcasr.utils.general import load_model, save_model, load_checkpoint, load_optimizer, get_model_class
from lcasr.utils.augmentation import SpecAugment
import resource
import time
from sentencepiece import SentencePieceProcessor 

from einops import rearrange
import numpy as np
import os
import wandb
from contextlib import nullcontext
from functools import partial

from torch.cuda.amp import GradScaler
from torch import autocast

from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
import random
random.seed(1234)


def blank_p(logits, tokenizer):
    lset = logits.detach().cpu()
    if torch.rand(1) < 0.05: # print 5 percent of the time
        print(tokenizer.decode([el for el in lset[0].argmax(dim=-1).tolist() if el != lset.shape[-1]-1]))
    lset = rearrange(lset, 'b n v -> (b n) v')
    lset_max = lset.argmax(dim=-1)
    lset_max = lset_max[lset_max == (lset.shape[-1]-1)]
    blank_p = lset_max.shape[0] / lset.shape[0]
    return blank_p


def backwards_pass(
        model:SCConformerXL,
        clip_value:float,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler,
        scaler:GradScaler,
    ):
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) if clip_value > 0 else None
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    if scheduler != None and scheduler.is_warmup:
        scheduler.step()


def apply_augmentation(audio, lengths, augmentation, epoch, start_augment_after_n_epochs, is_warmup):
    if start_augment_after_n_epochs == -1 or epoch < start_augment_after_n_epochs or not exists(augmentation) or is_warmup:
        return audio
    else:
        return augmentation(audio, lengths)
    
def get_dtype(dtype:str) -> torch.dtype:
    if dtype == 'bfloat16':
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16
    elif dtype == 'float32':
        return torch.float32
    else:
        raise ValueError(f'invalid dtype: {dtype}')

def run_benchmark(
        args:argparse.Namespace,
        model:torch.nn.Module, 
        device:torch.device,
        tokenizer:SentencePieceProcessor,
        optimizer:torch.optim.Optimizer,
    ):
    scaler = GradScaler() 
    clip_value = args.config['training'].get('clip_value', 0.8) 
    random.seed(args.config['training'].get('random_seed', 12345))
    dtype = get_dtype(args.config['training'].get('dtype', 'bfloat16'))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    model.train()

    model_dtype = next(model.parameters()).dtype
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')

    backprop_every, backwards_every = args.config['training']['backprop_every'], args.config['training'].get('backwards_every', 1)
    assert backprop_every >= backwards_every, f'backprop_every ({backprop_every}) must be >= backwards_every ({backwards_every})'
    
    batch_size = args.batch_size
    seq_len = args.seq_len

    steps = args.steps
    start_time = time.time()

    for i in range(steps):
        print(f'step {i}/{steps}')

        audio = torch.randn(batch_size, 80, seq_len)
        a_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        txt = torch.randint(1, tokenizer.vocab_size(), (batch_size, seq_len // 16))
        t_lengths = torch.full((batch_size,), seq_len // 16, dtype=torch.long)
 

        audio, a_lengths = audio.to(device, dtype=model_dtype), a_lengths.to(device)

        with autocast(device.type, dtype=dtype) if torch.cuda.is_available() else nullcontext():
            print(audio.shape)
            out = model(audio_signal = audio, length = a_lengths)
 
            cur_probs = out['final_posteriors']
            B,N,C = cur_probs.shape 
            loss = ctc_loss_fn(cur_probs.transpose(0,1), txt, out['length'], t_lengths).sum()
            


        
        # cur_tokens_in_loss += B * N
        cur_tokens_in_loss = (sum(a_lengths)) # total number of acoustic frames in batch

        scaler.scale(((loss) / (batch_size*seq_len)) * 100).backward() # divide by chunk*batch_size constant to weight smaller batches less


        full_loss = loss 
        full_loss /= cur_tokens_in_loss
        full_loss *= 100
        loss_to_log = full_loss.item()
        print(f'loss: {loss_to_log}')
        
        backwards_pass(
            model = model,
            clip_value = clip_value,
            optimizer = optimizer,
            scheduler = None,
            scaler = scaler
        )
        
        cur_tokens_in_loss, cur_loss = 0, torch.tensor(0.0, dtype=model_dtype, device=device)

        
    end_time = time.time()
    print(f'Total time (s): {end_time - start_time}')
    print(f'Average time per step (s): {(end_time - start_time) / steps}')
    avg_time_per_frame = (end_time - start_time) / (steps * batch_size * seq_len)
    print(f'Average time per frame (s): {avg_time_per_frame}')
    print(f'Frames per second: {1 / avg_time_per_frame}')

    return model
            
            


def main(args):
    args.config_path = args.config
    args.config = OmegaConf.load(args.config)


    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    # set random seed for initialization
    torch.manual_seed(12345), torch.cuda.manual_seed(12345)
    model = load_model(args.config, tokenizer.vocab_size(), get_model_class(config = args.config))
    tparams = model.print_total_params()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    optimizer, _ = load_optimizer(args.config, model, and_scheduler=False)

    model = model.to(device)

    run_benchmark(
        args = args, 
        model = model, 
        device = device,
        tokenizer=tokenizer,
        optimizer = optimizer, 
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-rm_sched', '--remove_scheduler', action='store_true', help='remove scheduler from checkpoint')
    parser.add_argument('-reset_step', '--reset_step', action='store_true', help='reset step to 0')
    parser.add_argument('-anomaly', '--anomaly', action='store_true', help='turn on anomaly detection')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-pin_memory', '--pin_memory', action='store_true', help='pin memory for dataloader')
    parser.add_argument('-prefetch', '--prefetch_factor', type=int, default=1, help='prefetch factor for dataloader')

    parser.add_argument('-debug_hooks', '--debug_hooks', action='store_true', help='add hooks to log gradient/activation info')

    parser.add_argument('-batch_size', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('-seq_len', '--seq_len', type=int, default=360000, help='sequence length')
    parser.add_argument('-steps', '--steps', type=int, default=100, help='number of steps to run')  

    args = parser.parse_args()


    main(args)
      