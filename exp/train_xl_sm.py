import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.dataloading import VariableBatchSimpleDataloader, chunk_spectogram, chunk_text_json
import traceback
from lcasr.utils.hooks import add_debug_backwards_hooks
from lcasr.utils.scheduling import CosineLRScheduler, SequenceWarmupManager
from lcasr.utils.helpers import exists

from lcasr.utils.general import load_model, save_model, load_checkpoint, load_optimizer
import resource

from einops import rearrange
import numpy as np
import os
import gc

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
    # print 20 percent of the time
    if torch.rand(1) < 0.05: # let's randomly decode and print utterances
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

    if scheduler.is_warmup:
        scheduler.step()



def train(
        args:argparse.Namespace,
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer,
        scheduler:CosineLRScheduler,
        sequence_scheduler:SequenceWarmupManager,
        device:torch.device,
        step:int = 0,
        seen_ids:List[str] = [],
    ):
    scaler = GradScaler()
            
    clip_value = args.config['training'].get('clip_value', 0.5) 
    intermediate_loss_weighting = args.config['training'].get('intermediate_loss_weighting', 0.0) # not used

    wandb_config = args.config['wandb']
    
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    model.train()
    model_dtype = next(model.parameters()).dtype
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')

    overlap = args.config.audio_chunking['overlap']
    if overlap > 0:
        raise NotImplementedError('Overlap during trainig not implemented (also might not be a good idea :P)')

    backprop_every = args.config['training']['backprop_every']
    backwards_every = args.config['training'].get('backwards_every', 1)
    assert backprop_every >= backwards_every, f'backprop_every ({backprop_every}) must be >= backwards_every ({backwards_every})'
    batch_size = args.config['training']['batch_size']

    max_cache_length = args.config['training']['max_seq_len']

    cur_tokens_in_loss = 0
    cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)

    tokenizer = dataloader.tokenizer
    chunk_size = args.config.audio_chunking['size']
    chunk_overlap = overlap

    if exists(sequence_scheduler):
        chunk_size = sequence_scheduler.cur_sequence_length
        batch_size = sequence_scheduler.cur_batch_size

    pad_id = tokenizer.pad_id()

    last_podcast = step
    cur_podcast = step
    podcasts_since_last_save = 0

    i = -1
    finished = False
    dataloader_iter = iter(dataloader)
    total_recordings = dataloader.total_recordings()
    pbar = tqdm(total = len(dataloader), desc = f'Training')

    while not finished:#################
        try:
            batch = next(dataloader_iter)
            i += 1
            pbar.update(1) if i > 0 else None
        except StopIteration:
            finished = True
            continue
        ################################
        audio, audio_length, txt, ids = batch
        seen_ids.extend(ids)
        assert audio.shape[0] == 1, 'Batch size must be 1 for this script, i.e 1 podcast at a time'
        cur_batch_size = 1
        ###############################
        cur_podcast += 1
        podcasts_since_last_save += (cur_podcast - last_podcast)
        if podcasts_since_last_save > args.config['checkpointing']['save_every_n_steps']:
            torch.cuda.empty_cache() 
            save_model(
                model = model, 
                optimizer = optimizer, 
                scheduler = scheduler, 
                podcast_step = cur_podcast, 
                config = args.config,
                sequence_scheduler = sequence_scheduler,
                seen_ids = seen_ids,
            )
            podcasts_since_last_save = 0
        last_podcast = cur_podcast
        ###############################
        
        
        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        max_len = max([el.shape[-1] for el in audio_chunks_])
        txt = " ".join([el['word'] for el in txt[0]])
        txt = torch.LongTensor(tokenizer.encode(txt))[None,:]
        txt_length = torch.LongTensor([txt.shape[-1]])

        ind_audio_lengths = []
       
        for i in range(len(audio_chunks_)):
            ind_audio_lengths.append(audio_chunks_[i].shape[-1])
            if audio_chunks_[i].shape[-1] < max_len:
                padding = torch.zeros(1, audio_chunks_[i].shape[1], max_len - audio_chunks_[i].shape[-1])
                audio_chunks_[i] = torch.cat([audio_chunks_[i], padding], dim=-1)
        ind_audio_lengths = torch.LongTensor(ind_audio_lengths)
        audio_chunks = torch.cat(audio_chunks_, dim=0)
   
        backwards_every_loss = 0.0
        steps_since_backwards = 0

        del audio, audio_chunks_
        chunks = []
      

        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                scheduler.set_cosine_schedule(total_recordings=total_recordings, cur_podcast=cur_podcast)


        ################################
 
        try:
            audio_chunks, ind_audio_lengths = audio_chunks.to(device, dtype=model_dtype), ind_audio_lengths.to(device)

            with autocast(device.type, dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext():
                out = model(
                    audio_signal = audio_chunks, 
                    length = ind_audio_lengths,
                )
                cur_probs = out['final_posteriors']
                
                cur_probs = rearrange(cur_probs, 'b n v -> 1 (b n) v')
                out_length = out['length'].sum()[None]
                print(out_length)
                cur_probs = cur_probs[:, :out_length[0], :]
             
                B,N,C = cur_probs.shape 
                loss = ctc_loss_fn(cur_probs.transpose(0,1), txt, out_length, txt_length).sum()
                
            blank_prob = blank_p(cur_probs.detach(), dataloader.tokenizer)
            # check for nan in loss
            if torch.isnan(loss):
                print('OH NO! NAN IN LOSS, SKIPPING') # TODO: set kv cache to None here
                continue

            cur_loss += loss

            backwards_every_loss += loss
            steps_since_backwards += 1
            
            # cur_tokens_in_loss += B * N
            cur_tokens_in_loss += (sum(ind_audio_lengths)) # total number of acoustic frames in batch

      
            scaler.scale(((backwards_every_loss) / (chunk_size*batch_size)*steps_since_backwards) * 100).backward() # divide by chunk*batch_size constant to weight smaller batches less
            steps_since_backwards = 0
            backwards_every_loss = 0

            full_loss = cur_loss 
            full_loss /= cur_tokens_in_loss
            full_loss *= 100
            loss_to_log = full_loss.item()
            print(f'loss: {full_loss}')
            
            backwards_pass(
                model = model,
                clip_value = clip_value,
                optimizer = optimizer,
                scheduler = scheduler,
                scaler = scaler
            )

            learning_rate = scheduler.get_last_lr()[0]

            if wandb_config['use']:
                wandb.log({
                    'loss': loss_to_log,
                    'blank_p': blank_prob,
                    'learning_rate': learning_rate,
                    'sequence_length': chunk_size,
                    'batch_size': batch_size,
                })
            
            cur_tokens_in_loss = 0
            cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)


        except RuntimeError as e: 
            if 'an illegal memory access was encountered' in str(e): 
                print(e,'\n --- skipping batch ---')
                continue
            else:
                print(traceback.format_exc()) 
                raise e

        if not scheduler.is_warmup: # step every batch
            scheduler.step(epoch = cur_podcast)

        if exists(sequence_scheduler):
            to_update, new_seq_len, new_bs = sequence_scheduler.step(steps = cur_batch_size)
            if to_update:
                args.config['audio_chunking']['size'] = new_seq_len
                chunk_size = new_seq_len
                batch_size = new_bs
                dataloader.update_batch_size(
                    batch_size = batch_size,
                    seen_ids = seen_ids,
                )
                if args.config['model']['use_rotary'] and args.config['sequence_scheduler'].get('interpolate_rotary', False):
                    model.rotary_pos_emb.rotary_interpolation_factor = model.rotary_pos_emb.rotary_interpolation_factor * sequence_scheduler.increase_by_multiplier
                dataloader_iter = iter(dataloader)
                pbar.total = len(dataloader) # update total of tqdm
                
        del chunks
        
    save_model( # save final model
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        podcast_step = cur_podcast,
        config = args.config,
        sequence_scheduler = sequence_scheduler,
        seen_ids = seen_ids,
    )
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
        run_name = None if 'name' not in wandb_config else wandb_config['name']

        wandb.init(project=project_name, config=args.config, name=run_name) if w_id == '' else wandb.init(project=project_name, id=w_id, resume="must", config=args.config, allow_val_change=True)
        wandb.watch(model, log="all") # sometimes this causes a crash ):
        wandb.config.update({'total_params': tparams}, allow_val_change=True)
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')

    model = model.to(device)
    optimizer, scheduler = load_optimizer(args.config, model)

    sequence_scheduler = None
    if 'sequence_scheduler' in args.config:
        sequence_scheduler = SequenceWarmupManager(
            initial_batch_size = args.config['training']['batch_size'],
            initial_sequence_length = args.config['audio_chunking']['size'],
            **args.config['sequence_scheduler']
        )

    seen_ids, step = load_checkpoint(
        args = args, 
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        sequence_scheduler = sequence_scheduler,
        path = args.config['checkpointing']['dir'],
        device = device
    )

    if args.reset_step:
        seen_ids, step = [], 0

    print(f'Starting from podcast: {len(seen_ids)}')
    # skip data up to step
    dataloader = VariableBatchSimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = 1,
        chunk_size = args.config.audio_chunking['size'],
        chunk_overlap = args.config.audio_chunking['overlap'],
        num_workers = args.num_workers,
        pin_memory = args.pin_memory,
        prefetch = args.prefetch_factor,
        seen_ids = seen_ids,
    )
    
    if args.debug_hooks:
        assert wandb_config['use'], 'must have wandb enabled when - arg.debug_hooks ==  True - to log debug hooks outputs'
        logger = partial(wandb.log, commit=False)
        add_debug_backwards_hooks(model = model, logger = logger)
    
    if sequence_scheduler and dataloader.batch_size != sequence_scheduler.cur_batch_size:
        print('WARNING: dataloader batch size does not match sequence scheduler batch size, updating dataloader batch size')
        dataloader.update_batch_size(batch_size = sequence_scheduler.cur_batch_size, seen_ids = seen_ids)

    final_model = train(
        args = args, 
        model = model, 
        dataloader = dataloader, 
        optimizer = optimizer, 
        scheduler = scheduler,
        sequence_scheduler = sequence_scheduler, 
        device = device, 
        seen_ids = seen_ids,
        step = step,
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-rm_sched', '--remove_scheduler', action='store_true', help='remove scheduler from checkpoint')
    parser.add_argument('-reset_step', '--reset_step', action='store_true', help='reset step to 0')
    parser.add_argument('-anomaly', '--anomaly', action='store_true', help='turn on anomaly detection')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-pin_memory', '--pin_memory', action='store_true', help='pin memory for dataloader')
    parser.add_argument('-prefetch', '--prefetch_factor', type=int, default=1, help='prefetch factor for dataloader')

    parser.add_argument('-debug_hooks', '--debug_hooks', action='store_true', help='add hooks to log gradient/activation info')

    args = parser.parse_args()


    main(args)
      