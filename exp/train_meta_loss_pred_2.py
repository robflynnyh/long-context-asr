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
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail

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
    ):
    
    clipset1 = [el for el in model.parameters()]

    torch.nn.utils.clip_grad_norm_(clipset1, clip_value) if clip_value > 0 else None

   
    
    optimizer.step()
   
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
    
def calc_wer(ctc_decoder, logits, lengths, targets):
    text = [ctc_decoder(logits[i, :lengths[i]]) for i in range(logits.shape[0])]
    wers = []
    for i in range(len(text)):
        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[text[i]], references=[targets[i]])
        wer = 0 if wer == np.inf else wer
        wers.append(wer*100)
    return torch.tensor(wers)

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
        epoch:int = 0,
        augmentation:SpecAugment = None,
    ):
    
    clip_value = args.config['training'].get('clip_value', 0.8) 
    random.seed(args.config['training'].get('random_seed', 12345))
    wandb_config = args.config['wandb']
    dtype = get_dtype(args.config['training'].get('dtype', 'bfloat16'))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    model.train()

    model_dtype = next(model.parameters()).dtype
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='none')

    backprop_every, backwards_every = args.config['training']['backprop_every'], args.config['training'].get('backwards_every', 1)
    assert backprop_every >= backwards_every, f'backprop_every ({backprop_every}) must be >= backwards_every ({backwards_every})'
    
    batch_size = args.config['training']['batch_size']
    max_cache_length = args.config['training'].get('max_cache_length', 0)

    

    cur_tokens_in_loss, cur_loss = 0, torch.tensor(0.0, dtype=model_dtype, device=device)

    tokenizer = dataloader.tokenizer
    chunk_size, chunk_overlap = args.config.audio_chunking['size'], 0 # previously args.config.audio_chunking['overlap'] though this is not used anymore

    if exists(sequence_scheduler):
        chunk_size = sequence_scheduler.cur_sequence_length
        batch_size = sequence_scheduler.cur_batch_size

    meta_loss_fn = lambda a, b: torch.norm(a - b, dim=1, p=2)
    nans_in_a_row = 0

    pad_id = tokenizer.pad_id()
    last_podcast, cur_podcast, podcasts_since_last_save = step, step, 0
    max_epochs = args.config['training'].get('max_epochs', 1)

    i, finished = -1, False
    dataloader_iter = iter(dataloader)
    total_recordings = dataloader.total_recordings() * max_epochs
    pbar = tqdm(total = len(dataloader), desc = f'Training - Epoch {epoch}')
    start_spec_augment_after_n_epochs = args.config['training'].get('start_spec_augment_after_n_epochs', -1)

    ctc_decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)

    while not finished:#################
        try:
            batch, i = next(dataloader_iter), i + 1
            pbar.update(1) if i > 0 else None
        except StopIteration:
            epoch += 1
            seen_ids = reset_seen_ids(seen_ids = seen_ids, epoch = epoch - 1)
            if epoch >= max_epochs:
                finished = True
            else:
                dataloader.update(
                    batch_size = dataloader.batch_size, 
                    seen_ids = seen_ids,
                    random_seed = random.randint(0, 10000),
                )
                dataloader_iter = iter(dataloader)
                pbar = tqdm(total = len(dataloader), desc = f'Training - Epoch {epoch}')
            continue
        ################################

        frames_per_batch = chunk_size * batch_size
        audio, audio_lengths, txt, ids = batch
        seen_ids.extend(ids)
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
                seen_ids = seen_ids,
                epoch = epoch,
            )
            podcasts_since_last_save = 0
        last_podcast = cur_podcast
        ###############################

       
        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = chunk_overlap)[:batch_size]
        audio_lengths = torch.LongTensor([el.shape[-1] for el in audio_chunks_]).to(device)
        max_length = audio_lengths.max()
        audio_chunks_ = [torch.nn.functional.pad(el, (0, max_length - el.shape[-1])) for el in audio_chunks_]
        audio_chunks_ = torch.cat(audio_chunks_, dim=0)
        txt_chunks_untokenized = chunk_text_json(text = txt[0], chunk_size = chunk_size, chunk_overlap = chunk_overlap, spectogram_length = audio.shape[-1])[:batch_size]
        txt_chunks_tokenized = [torch.LongTensor(tokenizer.encode(el)) for el in txt_chunks_untokenized]
        txt_lengths = torch.LongTensor([len(el) for el in txt_chunks_tokenized]).to(device)
        txt_chunks = torch.nn.utils.rnn.pad_sequence(txt_chunks_tokenized, batch_first=True, padding_value=pad_id).to(device)
        

        del audio
        backwards_every_loss, steps_since_backwards = 0.0, 0
        

        
        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                scheduler.set_cosine_schedule(total_recordings=total_recordings, cur_podcast=cur_podcast)
        prev_selection_mask, last_kv_set = None, None # selection mask from previous chunk
        ################################
 
        try:
    
            audio, a_lengths = audio_chunks_, audio_lengths
          

            txt, t_lengths = txt_chunks, txt_lengths
        
                

            audio, a_lengths = audio.to(device, dtype=model_dtype), a_lengths.to(device)

        
            audio = apply_augmentation(audio=audio, lengths=a_lengths, augmentation=augmentation, start_augment_after_n_epochs=start_spec_augment_after_n_epochs, epoch=epoch, is_warmup=scheduler.is_warmup)
            
            out = model(
                audio_signal = audio, 
                length = a_lengths,
            )
            
            cur_probs = out['final_posteriors']
            #meta_pred = out['meta_pred']
            B,N,C = cur_probs.shape 
            loss = ctc_loss_fn(cur_probs.transpose(0,1), txt, out['length'], t_lengths).sum()
            original_loss = ctc_loss_fn(model.original_probs.transpose(0,1), txt, out['length'], t_lengths).sum()

            blank_prob = blank_p(cur_probs.detach(), dataloader.tokenizer)
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


            cur_loss += loss
            
            # cur_tokens_in_loss += B * N
            cur_tokens_in_loss += (sum(a_lengths)) # total number of acoustic frames in batch

            #((loss)).backward(inputs=[model.reprs])
            repr_grads = torch.autograd.grad(original_loss, model.reprs, create_graph=False, retain_graph=False)[0].to(model.grad_pred.dtype).detach()
            #((loss / chunk_size*batch_size)*100).backward(retain_graph=True)
        
            meta_grad_pred = model.grad_pred
        
            #_,_,_=model.meta_decoder.vq(repr_grads)

            # norm 50% of the time
            # if torch.rand(1) < 0.5:
            #     repr_grads = repr_grads / (repr_grads.norm(dim=-1, keepdim=True) + 1e-8)
            #     meta_grad_pred = meta_grad_pred / (meta_grad_pred.norm(dim=-1, keepdim=True) + 1e-8)
            # #print(torch.nn.functional.cosine_similarity(repr_grads, meta_grad_pred, dim=-1))
            #meta_loss = ((((repr_grads - meta_grad_pred) ** 2).sum(-1)) ** 0.5).sum() / (batch_size*chunk_size)
            meta_grad_pred_1, meta_grad_pred_2 = meta_grad_pred, meta_grad_pred.detach()
            meta_loss_1 = meta_loss_fn(rearrange(repr_grads, 'b n v -> (b n) v'), rearrange(meta_grad_pred_1, 'b n v -> (b n) v')).sum() / (batch_size*chunk_size)
            meta_loss_2 = meta_loss_fn(rearrange(repr_grads[torch.randperm(repr_grads.shape[0])], 'b n v -> (b n) v'), rearrange(meta_grad_pred_2, 'b n v -> (b n) v')).sum() / (batch_size*chunk_size)
            
            #print(repr_grads.norm(), meta_grad_pred_1.norm(), meta_grad_pred_2.norm())
            
            #print(torch.linalg.norm(repr_grads, ord=2, dim=-1).min(), torch.linalg.norm(repr_grads, ord=2, dim=-1).max())
            # dot_p = torch.einsum('bnv,bmv->bnm', meta_grad_pred_1, repr_grads)
            # scores = dot_p.log_softmax(dim=-1)
            # # take identity and sum
            # scores = scores * torch.eye(scores.shape[-1])[None].to(scores.device)
            # scores = -scores.sum()
            

            cosim = (torch.nn.functional.cosine_similarity(repr_grads, meta_grad_pred_1, dim=-1) * -1 + 1)#.mean()

            #(cosim.sum() / chunk_size*batch_size).backward()#inputs=cosim_backwards_params)
            cosim = cosim.mean()

            
            ((loss / chunk_size*batch_size)*100).backward()#inputs=ctc_loss_params)

            
            #(meta_loss_1 *100).backward()
            # #print(repr_grads.dtype, meta_grad_pred.dtype)
            # meta_csim = (torch.nn.functional.cosine_similarity(repr_grads, meta_grad_pred, dim=-1) * -1 + 1).mean()
            # meta_mse = torch.nn.functional.mse_loss(repr_grads, meta_grad_pred, reduction='mean')
            # meta_loss = (meta_csim + meta_mse) / 2
            # scaler.scale(meta_loss).backward()
            
    
            full_loss = cur_loss 
            full_loss /= cur_tokens_in_loss
            full_loss *= 100
            loss_to_log = full_loss.item()

            original_loss = ((original_loss / cur_tokens_in_loss) * 100).item()
            print(f'loss: {loss_to_log}')
            
            backwards_pass(
                model = model,
                clip_value = clip_value,
                optimizer = optimizer,
                scheduler = scheduler,
             
            )
            learning_rate = scheduler.get_last_lr()[0]
            

            if wandb_config['use']:
                wandb.log({
                    # 'a_lengths': a_lengths.sum().item(),
                    # 't_lengths': t_lengths.sum().item(),
                    # 'bn': B*N,
                    #'scores': (scores / cur_tokens_in_loss).item(),
                    'cosim': cosim.item(),
                    'original_loss': original_loss,
                    'loss': loss_to_log,
                    'meta_loss_1': (meta_loss_1 / cur_tokens_in_loss).item(),
                    'meta_loss_2': (meta_loss_2 / cur_tokens_in_loss).item(),
                    'blank_p': blank_prob,
                    'learning_rate': learning_rate,
                    'sequence_length': chunk_size,
                    'batch_size': batch_size,
                    'epoch': epoch,
                    'spec_augment': int(True) if start_spec_augment_after_n_epochs != -1 and epoch >= start_spec_augment_after_n_epochs and scheduler.is_warmup == False else int(False),
                })
            
            cur_tokens_in_loss, cur_loss = 0, torch.tensor(0.0, dtype=model_dtype, device=device)
       

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
       
                if args.config['model']['use_rotary'] and args.config['sequence_scheduler'].get('interpolate_rotary', False):
                    model.rotary_pos_emb.rotary_interpolation_factor = model.rotary_pos_emb.rotary_interpolation_factor * sequence_scheduler.increase_by_multiplier
                dataloader_iter = iter(dataloader)
                pbar.total = len(dataloader) # update total of tqdm
                
        
    save_model( # save final model
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        podcast_step = cur_podcast,
        config = args.config,
        sequence_scheduler = sequence_scheduler,
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
    paired_data = lcasr.utils.audio_tools.load_json(args.config['data']['path'])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb_config = args.config['wandb']
    if wandb_config['use']:
        project_name, w_id = wandb_config['project_name'], wandb_config['id']
        run_name = None if 'name' not in wandb_config else wandb_config['name']
        wandb_dir = args.config['wandb'].get('dir', './')
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

    # for param in model.meta_layers_m.parameters():
    #     param.requires_grad = True

    # for param in model.magpred.parameters():
    #     param.requires_grad = True
    
    # for param in model.combine.parameters():
    #     param.requires_grad = True


    optimizer, scheduler = load_optimizer(args.config, model)

    sequence_scheduler = None
    if 'sequence_scheduler' in args.config:
        sequence_scheduler = SequenceWarmupManager(
            initial_batch_size = args.config['training']['batch_size'],
            initial_sequence_length = args.config['audio_chunking']['size'],
            **args.config['sequence_scheduler']
        )

    seen_ids, step, epoch = load_checkpoint(
        args = args, 
        model = model, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        sequence_scheduler = sequence_scheduler,
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
        random_seed = random_seed,
    )

    # None if start_spec_augment_after_n_epochs == -1 or epoch < start_spec_augment_after_n_epochs else 
    augmentation = SpecAugment(**args.config['spec_augment']) if 'spec_augment' in args.config else None
    assert exists(augmentation) or start_spec_augment_after_n_epochs == -1, 'must have spec augment in config if start_spec_augment_after_n_epochs > 0'

    if args.debug_hooks:
        assert wandb_config['use'], 'must have wandb enabled when - arg.debug_hooks ==  True - to log debug hooks outputs'
        logger = partial(wandb.log, commit=False)
        add_debug_backwards_hooks(model = model, logger = logger)
    


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
        augmentation = augmentation,
        epoch = epoch
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
      