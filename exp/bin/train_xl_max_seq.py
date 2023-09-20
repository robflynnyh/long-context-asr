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
    if torch.rand(1) < 0.05:
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
        skip_to:int = 0,
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

    last_podcast = skip_to
    cur_podcast = skip_to
    podcasts_since_last_save = 0

    i = -1
    finished = False
    dataloader_iter = iter(dataloader)
    total_recordings = dataloader.total_recordings()
    pbar = tqdm(total = len(dataloader), desc = f'Training')

    total_tokens = 0
    fback_times = []

    while not finished:#################
        try:
            batch = next(dataloader_iter)
            i += 1
            pbar.update(1) if i > 0 else None
        except StopIteration:
            finished = True
            continue
        ################################

        audio, audio_lengths, txt, _ = batch
        cur_batch_size = audio.shape[0]

        ###############################
        cur_podcast += audio.shape[0]
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
            )
            podcasts_since_last_save = 0
        last_podcast = cur_podcast
        ###############################
        
        
        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        txt_chunks = [chunk_text_json(text = el, chunk_size = chunk_size, chunk_overlap = chunk_overlap, spectogram_length = audio.shape[-1]) for el in txt]

        backwards_every_loss = 0.0
        steps_since_backwards = 0

        del audio
        chunks = []
        culm_lengths_audio = torch.zeros_like(audio_lengths)

        ################################
        for ix, el in enumerate(audio_chunks_):

            remove_mask = ~(culm_lengths_audio > audio_lengths)
            cur_chunks, cur_culm_lengths = el[remove_mask], culm_lengths_audio[remove_mask]
            cur_lengths = cur_chunks.shape[-1] - (cur_culm_lengths + cur_chunks.shape[-1] - audio_lengths[remove_mask] - chunk_overlap).clamp(0)
          
            enc_txt_chunks = [torch.LongTensor(tokenizer.encode(el[ix])) for i, el in enumerate(txt_chunks) if remove_mask[i]]
            enc_txt_chunks_lengths = torch.LongTensor([el.shape[0] for el in enc_txt_chunks])
            enc_txt_chunks = torch.nn.utils.rnn.pad_sequence(enc_txt_chunks, batch_first=True, padding_value=pad_id)
            if enc_txt_chunks_lengths.max() == 0:
                continue # skip if none contain text (bad batch)
            chunks.append({
                'audio':cur_chunks,
                'txt':enc_txt_chunks,
                'txt_lengths':enc_txt_chunks_lengths,
                'audio_lengths':cur_lengths,
                'selection_mask':remove_mask,
                'cur_culm_lengths':cur_culm_lengths,
            })

            culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (chunk_overlap if ix != 0 else 0)

        # # shuffle chunks if not using cache
        # if max_cache_length == 0:
        #     random.shuffle(chunks)

        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                scheduler.set_cosine_schedule(total_recordings=total_recordings, cur_podcast=cur_podcast)

        prev_selection_mask = None # selection mask from previous chunk
        last_kv_set = None
        ################################
        import time
        try:
            for ix, chunk_json in enumerate(chunks):
                print(f'chunk {ix}/{len(chunks)}')
               
                #audio, a_lengths = chunk_json['audio'], chunk_json['audio_lengths']
                audio = torch.randn(1, 80, chunk_size)
                a_lengths = torch.LongTensor([chunk_size])
                txt, t_lengths = chunk_json['txt'], chunk_json['txt_lengths']
                selection_mask = chunk_json['selection_mask']

                cur_selection_mask = None
                if prev_selection_mask != None and not torch.allclose(selection_mask, prev_selection_mask):
                    cur_selection_mask = selection_mask[prev_selection_mask]
                    

                audio, a_lengths = audio.to(device, dtype=model_dtype), a_lengths.to(device)
                print(a_lengths, audio.shape)
                stime = time.time()

                with autocast(device.type, dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext():
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
                    
                    cur_probs = out['final_posteriors']
                    B,N,C = cur_probs.shape 
                    loss = ctc_loss_fn(cur_probs.transpose(0,1), txt, out['length'], t_lengths).sum()
                    
                blank_prob = blank_p(cur_probs.detach(), dataloader.tokenizer)
                cur_loss += loss

                backwards_every_loss += loss
                steps_since_backwards += 1
                
                # cur_tokens_in_loss += B * N
                cur_tokens_in_loss += (sum(a_lengths)) # total number of acoustic frames in batch

                if (ix+1) % backwards_every == 0 or (ix+1) == len(chunks):
                    scaler.scale(((backwards_every_loss) / (chunk_size*batch_size)*steps_since_backwards) * 100).backward() # divide by chunk*batch_size constant to weight smaller batches less
                    last_kv_set.detach_() if last_kv_set != None else None
                    steps_since_backwards = 0
                    backwards_every_loss = 0

                if sum(a_lengths) == chunk_size:
                    etime = time.time()
                    total_tokens += (sum(a_lengths))
                    fback_times.append(etime-stime)
                    print(f'tokens per second: {total_tokens / sum(fback_times)}')

                if (ix+1) % backprop_every == 0 or (ix+1) == len(chunks): 
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

                prev_selection_mask = selection_mask.clone()

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
                    skip_to = cur_podcast,
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

    step = load_checkpoint(
        args = args, 
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        sequence_scheduler = sequence_scheduler,
        path = args.config['checkpointing']['dir'],
        device = device
    )

    if args.reset_step:
        step = 0

    print(f'Starting from podcast: {step}')
    # skip data up to step
    dataloader = VariableBatchSimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.config['training']['batch_size'],
        skip_to = step,
        chunk_size = args.config.audio_chunking['size'],
        chunk_overlap = args.config.audio_chunking['overlap'],
        num_workers = args.num_workers,
        pin_memory = args.pin_memory,
        prefetch = args.prefetch_factor,
    )
    
    if args.debug_hooks:
        assert wandb_config['use'], 'must have wandb enabled when - arg.debug_hooks ==  True - to log debug hooks outputs'
        logger = partial(wandb.log, commit=False)
        add_debug_backwards_hooks(model = model, logger = logger)
    
    if sequence_scheduler and dataloader.batch_size != sequence_scheduler.cur_batch_size:
        print('WARNING: dataloader batch size does not match sequence scheduler batch size, updating dataloader batch size')
        dataloader.update_batch_size(batch_size = sequence_scheduler.cur_batch_size, skip_to = step)

    final_model = train(
        args = args, 
        model = model, 
        dataloader = dataloader, 
        optimizer = optimizer, 
        scheduler = scheduler,
        sequence_scheduler = sequence_scheduler, 
        device = device, 
        skip_to = step
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

    args = parser.parse_args()


    main(args)
      