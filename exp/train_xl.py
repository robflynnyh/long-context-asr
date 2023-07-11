import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.dataloading import SimpleDataloader
import traceback

from lcasr.utils.general import load_model, save_model, load_checkpoint

from einops import rearrange
import numpy as np
import os

import madgrad
import wandb

from torch.cuda.amp import GradScaler
from torch import autocast

from apex.optimizers import FusedAdam
from torch.optim import Adam

import warnings
from torch.optim.lr_scheduler import _enable_get_lr_call

class CosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, peak_value, final_value):
        self.is_warmup = True
        self.warmup_steps = warmup_steps
        self.peak_value = peak_value
        self.final_value = final_value
        super().__init__(optimizer)
        
    def is_warming_up(self):
        if self.is_warmup:
            return self.last_epoch < self.warmup_steps
        else:
            return False

    def set_cosine_schedule(self, remaining_steps):
        # reset the step to 0
        self.last_epoch = 0
        self.is_warmup = False
        self.steps = remaining_steps

    def get_lr(self):
        if self.is_warmup:
            return [self.peak_value * min(1.0, self.last_epoch / self.warmup_steps) for _ in self.base_lrs]
        else:
            return [self.final_value + 0.5 * (self.peak_value - self.final_value) * (1 + np.cos((self.last_epoch) / (self.steps) * np.pi)) for _ in self.base_lrs]


def load_optimizer(config:Dict, model:torch.nn.Module):
    # check device
    model_device = next(model.parameters()).device.type

    optim_type = config['optimizer']['name']
    allowed_types = ['adam', 'madgrad']
    
    assert optim_type in allowed_types, f'Unknown optimizer {optim_type}, must be one of {allowed_types}'
    assert model_device in ['cpu', 'cuda'], f'Unknown device {model_device}, must be one of [cpu, cuda]'

    optim_args = config['optimizer']['args']

    if optim_type == 'adam':
        optimizer = Adam(model.parameters(), **optim_args) if model_device == 'cpu' else FusedAdam(model.parameters(), **optim_args)
    elif optim_type == 'madgrad':
        optimizer = madgrad.MADGRAD(model.parameters(), **optim_args)

    sheduler = CosineLRScheduler(
        optimizer = optimizer,
        warmup_steps = config['scheduler']['warmup_steps'],
        peak_value = config['optimizer']['args']['lr'],
        final_value = 0.0, # decay to 0
    )

    return optimizer, sheduler

def blank_p(logits, tokenizer):
    lset = logits.detach().cpu()
    # print 20 percent of the time
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

    if scheduler.is_warmup:
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
    ds_overlap = overlap // model.subsampling_factor
    backprop_every = args.config['training']['backprop_every']

    max_cache_length = args.config['training']['max_seq_len']

    cur_tokens_in_loss = 0
    cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)



    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        # save every 100 steps
        if i % args.config['checkpointing']['save_every_n_steps'] == 0 and i != 0:
            save_model(model, optimizer, scheduler, i*args.config['training']['batch_size'] + skip_to, args.config)

        chunks = batch['chunks']

        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                current_recording = i * args.config['training']['batch_size'] 
                total_recordings = len(dataloader) * args.config['training']['batch_size'] 
                remaining_recordings = total_recordings - current_recording
                remaining_steps = remaining_recordings // args.config['training']['batch_size']
                scheduler.set_cosine_schedule(remaining_steps)

        prev_selection_mask = None # selection mask from previous chunk
        last_kv_set = None
 
        try:
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
                    
                    if max_cache_length != 0:
                        out_kvs = out['kvs_to_cache'].clone()
                        last_kv_set = out_kvs[:, -max_cache_length:].clone()
                    
                
                    # check for nan
                    cur_probs = out['final_posteriors']
                    
                    B,N,C = cur_probs.shape 
                    
                    loss = ctc_loss_fn(cur_probs.transpose(0,1), txt, out['length'], t_lengths).sum()
                

                blank_prob = blank_p(cur_probs.detach(), dataloader.tokenizer)
                cur_loss += loss
    
                
                # cur_tokens_in_loss += B * N
                cur_tokens_in_loss += (sum(a_lengths)) # total number of acoustic frames in batch

                scaler.scale(((loss) / (sum(a_lengths))) * 100).backward()
                last_kv_set.detach_() if last_kv_set != None else None


                if (ix+1) % backprop_every == 0:
                    full_loss = cur_loss 
                    full_loss /= cur_tokens_in_loss
                    full_loss *= 100
                    loss_to_log = full_loss.item()
                    print(f'loss: {full_loss}')
                    backwards_pass(model, full_loss, optimizer, scheduler, scaler)

                    learning_rate = scheduler.get_last_lr()[0]

                    if wandb_config['use']:
                        wandb.log({
                            'loss': loss_to_log,
                            'blank_p': blank_prob,
                            'overlap_interp_factor_logits': model.overlap_interp_factor_logits.item(),
                            'learning_rate': learning_rate,
                        })

                    
                    cur_tokens_in_loss = 0
                    cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)

                prev_selection_mask = selection_mask.clone()

        except RuntimeError as e: # illegal mem access sometimes happening with flash attention.. 0: 
            if 'an illegal memory access was encountered' in str(e): 
                print(e,'\n --- skipping batch ---')
                continue
            else:
                print(traceback.format_exc()) 
                raise e

        if not scheduler.is_warmup: # step every batch
            scheduler.step()

    # save final model
    save_model(model, optimizer, None, i*args.config['training']['batch_size'] + skip_to, args.config)
    return model
            


def main(args):
    args.config = OmegaConf.load(args.config)

    checkpoint_dir = args.config['checkpointing']['dir']
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir); print(f'created checkpoint dir: {checkpoint_dir}')

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    paired_data = lcasr.utils.audio_tools.load_json(args.config['data']['path'])


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
    step = load_checkpoint(model, optimizer, scheduler, args.config['checkpointing']['dir'])
    print(f'Starting from podcast: {step}')
    # skip data up to step
    dataloader = SimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.config['training']['batch_size'],
        skip_to = step,
        chunk_size = args.config.audio_chunking['size'],
        chunk_overlap = args.config.audio_chunking['overlap'],
    )
    
    final_model = train(args, model, dataloader, optimizer, scheduler, device, skip_to = step)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')

    args = parser.parse_args()


    main(args)
    