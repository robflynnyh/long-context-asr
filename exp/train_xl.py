import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.audio_tools import SimpleDataloader

from lcasr.utils.general import load_model, save_model, load_checkpoint
from einops import rearrange
import os


import wandb

from torch.cuda.amp import GradScaler
from torch import autocast

from apex.optimizers import FusedAdam
from torch.optim import Adam

import warnings
from torch.optim.lr_scheduler import _enable_get_lr_call



def load_optimizer(config:Dict, model:torch.nn.Module):
    # check device

    def warmup(current_step: int):
        if current_step < config['scheduler']['warmup_steps']:
            return current_step / config['scheduler']['warmup_steps']
        else:
            return 1.0

    model_device = next(model.parameters()).device.type
    if model_device == 'cpu':
        optimizer = Adam(model.parameters(), **config.optimizer)
    elif model_device == 'cuda':
        print('-- Using FusedAdam --')
        optimizer = FusedAdam(model.parameters(), **config.optimizer)
    else:
        raise ValueError('Unknown device')

    sheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup)

    return optimizer, sheduler

def blank_p(logits, tokenizer):
    lset = logits.detach().cpu()
    # print 10 percent of the time
    if torch.rand(1) < 0.2:
        print(tokenizer.decode([el for el in lset[0].argmax(dim=-1).tolist() if el != lset.shape[-1]-1]))
    lset = rearrange(lset, 'b n v -> (b n) v')
    lset_max = lset.argmax(dim=-1)
    lset_max = lset_max[lset_max == (lset.shape[-1]-1)]
    
    blank_p = lset_max.shape[0] / lset.shape[0]
    # lset = torch.exp(lset)
    # blank_p = lset[:, -1].mean().item()
    return blank_p

def backwards_pass(
        model:SCConformerXL,
        loss:torch.Tensor,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler,
        scaler:GradScaler,
    ):
    
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()

    scheduler.step()



def train(
        args:argparse.Namespace,
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler, 
        device:torch.device,
        skip_to:int = 0,
    ):
    scaler = GradScaler()

    wandb_config = args.config['wandb']

    model.train()
    model_dtype = next(model.parameters()).dtype
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')

    overlap = args.config.audio_chunking['overlap']
    ds_overlap = overlap // 4 # 4 is the subsampling factor
    backprop_every = args.backprop_every


    cur_tokens_in_loss = 0
    
    cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)
    pbar = tqdm(dataloader)
    for ix, batch in enumerate(pbar):
        # save every 100 steps
        if ix % 100 == 0:
            save_model(model, optimizer, None, ix*args.batch_size + skip_to, args.config, f"model.pt")

        chunks = batch['chunks']


        # shuffle chunks
        #chunks = [chunks[i] for i in torch.randperm(len(chunks))]

        last_prob_set = None # last set of probabilities output by model
        last_kv_set = None # [KV (2), B, L, N, H, D]
        prev_selection_mask = None # selection mask from previous chunk
        # shuffle chunks
        #chunks = [chunks[i] for i in torch.randperm(len(chunks))]
        for ix, chunk_json in enumerate(chunks):
            print(f'chunk {ix}/{len(chunks)}')
            
            audio, a_lengths = chunk_json['audio'], chunk_json['audio_lengths']
            txt, t_lengths = chunk_json['txt'], chunk_json['txt_lengths']
            selection_mask = chunk_json['selection_mask']

            cur_selection_mask = None
            if prev_selection_mask != None and not torch.allclose(selection_mask, prev_selection_mask):
                cur_selection_mask = selection_mask[prev_selection_mask]
                

            audio, a_lengths = audio.to(device, dtype=model_dtype), a_lengths.to(device)

            with autocast(device.type, dtype=torch.bfloat16):
                cached_kvs = last_kv_set.clone() if last_kv_set != None else None
                cached_kv_lengths = torch.LongTensor([cached_kvs.shape[1]] * cached_kvs.shape[0]).to(device) if cached_kvs != None else None

                if cur_selection_mask != None and cached_kvs != None:
                    cached_kvs = cached_kvs[cur_selection_mask]
                    cached_kv_lengths = cached_kv_lengths[cur_selection_mask]
                
                out = model(
                    audio_signal = audio, 
                    length = a_lengths, 
                    cached_kvs = cached_kvs, 
                    cached_kv_lengths = cached_kv_lengths
                )
                
                if args.max_seq_len != 0:
                    out_kvs = out['kvs_to_cache'].clone()
                    if last_kv_set != None and ds_overlap != 0:
                        if cur_selection_mask != None:
                            last_kv_set = last_kv_set[cur_selection_mask]
                        interp_factor = model.overlap_interp_factor_kvs
                        out_kvs[:,:ds_overlap] *= interp_factor
                        o_len = out_kvs[:,:ds_overlap].shape[1]
                        o_len = min(o_len, ds_overlap)
                        out_kvs[:,:ds_overlap] += (1-interp_factor) * last_kv_set[:, -o_len:]
                    last_kv_set = out_kvs[:, -args.max_seq_len:].clone()
            
                # check for nan
                cur_probs = out['final_posteriors'].clone()
                B,N,C = cur_probs.shape 

                if last_prob_set != None and ds_overlap != 0:
                    if cur_selection_mask != None:
                        last_prob_set = last_prob_set[cur_selection_mask]
                    interp_factor = model.overlap_interp_factor_logits
                    cur_probs[:,:ds_overlap] *= interp_factor
                    o_len = cur_probs[:,:ds_overlap].shape[1]
                    o_len = min(o_len, ds_overlap)
                    cur_probs[:,:ds_overlap] += (1-interp_factor) * last_prob_set[:, -o_len:]

                last_prob_set = out['final_posteriors'].clone()
                loss = ctc_loss_fn(cur_probs.transpose(0,1), txt, out['length'], t_lengths)

            cur_loss += loss
            
            # cur_tokens_in_loss += B * N
            cur_tokens_in_loss += (sum(a_lengths)) # total number of acoustic frames in batch

            scaler.scale((loss / (sum(a_lengths))) * 100).backward()
  

            if cur_tokens_in_loss > backprop_every: # or ix == len(chunks) - 1:
                cur_loss /= cur_tokens_in_loss
                cur_loss *= 100
                loss_to_log = cur_loss.item() 
                print(f'loss: {loss_to_log}')
                backwards_pass(model, cur_loss, optimizer, scheduler, scaler)

                learning_rate = scheduler.get_last_lr()[0]

                if wandb_config['use']:
                    wandb.log({
                        'loss': loss_to_log,
                        'blank_p': blank_p(last_prob_set, dataloader.tokenizer),
                        'overlap_interp_factor_logits': model.overlap_interp_factor_logits.item(),
                        'learning_rate': learning_rate,
                    })

                
                cur_tokens_in_loss = 0
                last_prob_set.detach_() 
                last_kv_set.detach_() if last_kv_set != None else None
                cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)

            prev_selection_mask = selection_mask.clone()
            


def main(args):
    args.config = OmegaConf.load(args.config)
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    paired_data = lcasr.utils.audio_tools.load_json('/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb_config = args.config['wandb']
    if wandb_config['use']:
        project_name, w_id = wandb_config['project_name'], wandb_config['id']
        wandb.init(project=project_name, config=args.config) if w_id == '' else wandb.init(project=project_name, id=w_id, resume="must", config=args.config, allow_val_change=True)
        wandb.watch(model, log="all")
        wandb.config.update({'total_params': tparams}, allow_val_change=True)
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')

    model = model.to(device)
    optimizer, scheduler = load_optimizer(args.config, model)
    step = load_checkpoint(model, optimizer) if os.path.exists('model.pt') else 0
    print(f'Starting from podcast: {step}')
    
    # skip data up to step
    dataloader = SimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.batch_size,
        skip_to = step,
        chunk_size = args.config.audio_chunking['size'],
        chunk_overlap = args.config.audio_chunking['overlap'],
    )
    
    train(args, model, dataloader, optimizer, scheduler, device, skip_to = step)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-b', '--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('-bprop', '--backprop_every', type=int, default=1, help='backprop every n tokens')
    parser.add_argument('-max_seq', '--max_seq_len', type=int, default=0, help='max sequence length')
    
    args = parser.parse_args()
    main(args)
    