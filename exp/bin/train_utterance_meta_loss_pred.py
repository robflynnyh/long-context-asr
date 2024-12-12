import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
import traceback
from lcasr.utils.dataloading import Utterance_Dataloader, reset_seen_ids
from lcasr.utils.hooks import add_debug_backwards_hooks
from lcasr.utils.scheduling import CosineLRScheduler
from lcasr.utils.helpers import exists
from lcasr.utils.general import load_model, save_model, load_checkpoint, load_optimizer, get_model_class
from lcasr.utils.augmentation import SpecAugment
import resource
import time
import sentencepiece as spm

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
    
    scaler.unscale_(optimizer) if exists(scaler) else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) if clip_value > 0 else None
    scaler.step(optimizer) if exists(scaler) else optimizer.step()
    scaler.update() if exists(scaler) else None
    optimizer.zero_grad() 

    if scheduler.is_warmup:
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

def train(
        args:argparse.Namespace,
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer,
        scheduler:CosineLRScheduler,
        device:torch.device,
        tokenizer:spm.SentencePieceProcessor,
        step:int = 0,
        seen_ids:List[str] = [],
        epoch:int = 0,
        augmentation:SpecAugment = None,
        chunk_size=2048,
    ):
    #scaler = GradScaler() 
    clip_value = args.config['training'].get('clip_value', 0.8) 
    random.seed(args.config['training'].get('random_seed', 12345))
    wandb_config = args.config['wandb']
    dtype = get_dtype(args.config['training'].get('dtype', 'bfloat16'))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    if args.config['training'].get('loss', 'l2') == 'l2':
        meta_loss_fn = lambda a, b, d: (torch.norm(a - b, dim=1, p=2)).sum() / d
    elif args.config['training'].get('loss', 'mse') == 'mse':
        meta_loss_fn = lambda a, b, d: (torch.nn.functional.mse_loss(a, b, reduction='sum') / d)
    elif args.config['training'].get('loss', 'l2') == 'cosine':
        meta_loss_fn = lambda a, b, d: ((torch.nn.functional.cosine_similarity(a, b, dim=1) * -1 + 1)).mean()
    model.train()

    model_dtype = next(model.parameters()).dtype
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    
    batch_size = args.config['training']['batch_size']

    cur_tokens_in_loss, cur_loss = 0, torch.tensor(0.0, dtype=model_dtype, device=device)

    last_podcast, cur_podcast, podcasts_since_last_save = step, step, 0
    max_epochs = args.config['training'].get('max_epochs', 1)

    i, finished = -1, False
    total_recordings = dataloader.total_recordings() * max_epochs
    start_spec_augment_after_n_epochs = args.config['training'].get('start_spec_augment_after_n_epochs', -1)
    pbar = tqdm(total = len(dataloader), desc = f'Training')

    for epoch in range(epoch, max_epochs):
        pbar.set_description(f'Training - Epoch: {epoch}')
        for batch in dataloader:
            i += 1
            audio, audio_lengths, text, text_lengths, ids = batch['audio'], batch['audio_lengths'], batch['text'], batch['text_lengths'], batch['ids']

            cur_podcast += audio.shape[0]
            pbar.update(audio.shape[0])
            podcasts_since_last_save += (cur_podcast - last_podcast)
            print(podcasts_since_last_save, args.config['checkpointing']['save_every_n_steps'])
            if podcasts_since_last_save > args.config['checkpointing']['save_every_n_steps']:
                print('SAVING MODEL')
                torch.cuda.empty_cache() 
                save_model(
                    model = model, 
                    optimizer = optimizer, 
                    scheduler = scheduler, 
                    podcast_step = cur_podcast, 
                    config = args.config,
                    seen_ids = seen_ids,
                    epoch = epoch,
                )
                podcasts_since_last_save = 0
            last_podcast = cur_podcast
            seen_ids.extend(ids)

            was_warmup = scheduler.is_warmup
            if was_warmup:
                scheduler.is_warmup = scheduler.is_warming_up()
                if not scheduler.is_warmup and was_warmup:
                    scheduler.set_cosine_schedule(total_recordings=total_recordings, cur_podcast=cur_podcast)

            audio = apply_augmentation(audio=audio, lengths=audio_lengths, augmentation=augmentation, start_augment_after_n_epochs=start_spec_augment_after_n_epochs, epoch=epoch, is_warmup=scheduler.is_warmup)
            audio, audio_lengths = audio.to(device, dtype=model_dtype), audio_lengths.to(device)
            text, text_lengths = text.to(device), text_lengths.to(device)

            out = model(
                audio_signal = audio, 
                length = audio_lengths,
            )
    
            cur_probs = out['final_posteriors']
            B,N,C = cur_probs.shape 
            loss = ctc_loss_fn(cur_probs.transpose(0,1), text, out['length'], text_lengths).sum()
            original_loss = ctc_loss_fn(model.original_probs.transpose(0,1), text, out['length'], text_lengths).sum()
                    
            blank_prob = blank_p(cur_probs.detach(), tokenizer)
            # check for nan in loss
            if torch.isnan(loss):
                print('OH NO! NAN IN LOSS, SKIPPING') # TODO: set kv cache to None here
                wandb.log({'nan':True}) if wandb_config['use'] else None
                optimizer.zero_grad() # clear gradients
                nans_in_a_row += 1
                if nans_in_a_row > 100:
                    print('100 NANS in a row, exiting......')
                    exit()
                continue
            else:
                nans_in_a_row = 0



        
            
            # cur_tokens_in_loss += B * N
            cur_tokens_in_loss += (sum(audio_lengths)) # total number of acoustic frames in batch

            repr_grads = torch.autograd.grad(original_loss, model.reprs, create_graph=False, retain_graph=True)

            repr_grads = torch.cat(repr_grads, dim=0).to(model.grad_pred.dtype).detach()
        
            meta_grad_pred = model.grad_pred
            meta_grad_pred_1, meta_grad_pred_2 = meta_grad_pred, meta_grad_pred.detach()
            meta_loss_1 = meta_loss_fn(rearrange(repr_grads, 'b n v -> (b n) v'), rearrange(meta_grad_pred_1, 'b n v -> (b n) v'), d=(batch_size*chunk_size*6))
            meta_loss_2 = meta_loss_fn(rearrange(repr_grads[torch.randperm(repr_grads.shape[0])], 'b n v -> (b n) v'), rearrange(meta_grad_pred_2, 'b n v -> (b n) v'), d=(batch_size*chunk_size*6))
            cosim = (torch.nn.functional.cosine_similarity(repr_grads, meta_grad_pred_1, dim=-1) * -1 + 1).mean()
            #_,_,_=model.meta_decoder.v_bank(repr_grads)

            
            #(((loss) / (chunk_size*batch_size)) * 100).backward() # divide by chunk*batch_size constant to weight smaller batches less
            (meta_loss_1).backward()
            

            full_loss = loss 
            full_loss /= cur_tokens_in_loss
            full_loss *= 100
            loss_to_log = full_loss.item()
            print(f'loss: {full_loss}')
            
            backwards_pass(
                model = model,
                clip_value = clip_value,
                optimizer = optimizer,
                scheduler = scheduler,
                scaler = None
            )
            learning_rate = scheduler.get_last_lr()[0]
            

            if wandb_config['use']:
                wandb.log({
                    'meta_loss_1': meta_loss_1.item(),
                    'meta_loss_2': meta_loss_2.item(),
                    'cosim': (cosim).item(),
                    'original_loss': ((original_loss / cur_tokens_in_loss) * 100).item(),
                    'loss': loss_to_log,
                    'blank_p': blank_prob,
                    'learning_rate': learning_rate,
                    'sequence_length': chunk_size,
                    'batch_size': batch_size,
                    'epoch': epoch,
                    'spec_augment': int(True) if start_spec_augment_after_n_epochs != -1 and epoch >= start_spec_augment_after_n_epochs and scheduler.is_warmup == False else int(False),
                })
            
            cur_tokens_in_loss, cur_loss = 0, torch.tensor(0.0, dtype=model_dtype, device=device)

            if not scheduler.is_warmup: # step every batch
                scheduler.step(epoch = cur_podcast)

        # end of epoch
        print(f'End of epoch: {epoch}')
        torch.cuda.empty_cache()
        pbar.reset()
        pbar.refresh()
        reset_seen_ids(seen_ids = seen_ids, epoch = epoch - 1)
    
                

        
     


    save_model( # save final model
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        podcast_step = cur_podcast,
        config = args.config,
        seen_ids = seen_ids,
        epoch = epoch,
    )
    return model
            
            


def main(args):
    args.config_path = args.config
    args.config = OmegaConf.load(args.config)

    checkpoint_dir = args.config['checkpointing']['dir']
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir); print(f'created checkpoint dir: {checkpoint_dir}')

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    # set random seed for initialization
    torch.manual_seed(12345), torch.cuda.manual_seed(12345)
    model = load_model(args.config, tokenizer.vocab_size(), get_model_class(config = args.config))
    tparams = model.print_total_params()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb_config = args.config['wandb']
    if wandb_config['use']:
        project_name, w_id = wandb_config['project_name'], wandb_config['id']
        run_name = None if 'name' not in wandb_config else wandb_config['name']
        wandb_dir = args.config['wandb'].get('dir', './wandb')
        wandb.init(project=project_name, config=args.config, name=run_name, dir=wandb_dir) if w_id == '' else wandb.init(project=project_name, id=w_id, resume="must", config=args.config, allow_val_change=True, dir=wandb_dir)
        wandb.watch(model, log="all") # sometimes this causes a crash ):
        wandb.config.update({'total_params': tparams}, allow_val_change=True)
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')
        args.config['wandb']['id'] = wandb.run.id # add wandb config to args.config
        if wandb_config.get('update_config_with_wandb_id', False): OmegaConf.save(config=args.config, f=args.config_path)

    model = model.to(device)


    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.meta_decoder.parameters():
        param.requires_grad = True

    for param in model.meta_layers.parameters():
        param.requires_grad = True

    optimizer, scheduler = load_optimizer(args.config, model)


    seen_ids, step, epoch = load_checkpoint(
        args = args, 
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        path = args.config['checkpointing']['dir'],
        device = device
    )
    if args.reset_step:
        seen_ids, step, epoch = [], 0, 0 

    print(f'Starting from podcast: {len(seen_ids)}')
    random_seed = args.config['training'].get('random_seed', 1234)
    if random_seed == 'random':  # generate using time
        random_seed = int(time.time()) % 10000 
        print(f'random seed: {random_seed}')

    random.seed(random_seed)
    start_spec_augment_after_n_epochs = args.config['training'].get('start_spec_augment_after_n_epochs', -1)

    # skip data up to step
    dataloader = Utterance_Dataloader(
        utterance_folder = args.config['data']['path'],
        batch_size = args.config['training']['batch_size'],
        num_workers = args.config['training'].get('num_workers', 0),
        pin_memory = args.config['training'].get('pin_memory', False),
        prefetch = args.config['training'].get('prefetch_factor', None),
        seen_ids = seen_ids,
        random_seed = random_seed,
    )

    # None if start_spec_augment_after_n_epochs == -1 or epoch < start_spec_augment_after_n_epochs else 
    augmentation = SpecAugment(**args.config['spec_augment']) if 'spec_augment' in args.config else None
    assert exists(augmentation) or start_spec_augment_after_n_epochs == -1, 'must have spec augment in config if start_spec_augment_after_n_epochs > 0'

    if args.debug_hooks:
        assert wandb_config['use'], 'must have wandb enabled when - arg.debug_hooks ==  True - to log debug hooks outputs'
        add_debug_backwards_hooks(model = model, logger = partial(wandb.log, commit=False))
    
 
    final_model = train(
        args = args, 
        model = model, 
        dataloader = dataloader, 
        optimizer = optimizer, 
        scheduler = scheduler,
        device = device, 
        seen_ids = seen_ids,
        step = step,
        augmentation = augmentation,
        epoch = epoch,
        tokenizer = tokenizer
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-rm_sched', '--remove_scheduler', action='store_true', help='remove scheduler from checkpoint')
    parser.add_argument('-reset_step', '--reset_step', action='store_true', help='reset step to 0')
    parser.add_argument('-anomaly', '--anomaly', action='store_true', help='turn on anomaly detection')
    parser.add_argument('-num_workers', '--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-pin_memory', '--pin_memory', action='store_true', help='pin memory for dataloader')
    parser.add_argument('-prefetch', '--prefetch', type=int, default=None, help='prefetch factor for dataloader')

    parser.add_argument('-debug_hooks', '--debug_hooks', action='store_true', help='add hooks to log gradient/activation info')

    args = parser.parse_args()


    main(args)
      