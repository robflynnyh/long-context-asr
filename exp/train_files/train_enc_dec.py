import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
import traceback
from lcasr.utils.dataloading import VariableBatchSimpleDataloader, chunk_spectogram, chunk_text_json, reset_seen_ids
from lcasr.utils.hooks import add_debug_backwards_hooks
from lcasr.utils.scheduling import CosineLRScheduler, SequenceWarmupManager, RandomSequenceLengthManager
from lcasr.utils.helpers import exists
from lcasr.utils.general import load_model, save_model, load_checkpoint, load_optimizer, get_model_class, KeepCount, find_latest_checkpoint
from lcasr.utils.augmentation import SpecAugment
import resource
from lcasr.decoding.greedy import GreedyCTCDecoder

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


def resolve_checkpoint_path(path: str) -> str:
    if os.path.isdir(path):
        latest = find_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"no .pt checkpoints found in {path}")
        return os.path.join(path, latest)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def remap_legacy_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.endswith(".norm.scale"):
            new_key = new_key[: -len(".scale")] + ".weight"
        elif new_key.endswith(".out_proj.0.scale"):
            new_key = new_key[: -len(".scale")] + ".weight"
        remapped[new_key] = value
    return remapped


def load_pretrained_model_state(model: torch.nn.Module, path: str, device: torch.device) -> None:
    checkpoint_path = resolve_checkpoint_path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    try:
        model.load_state_dict(remap_legacy_state_dict_keys(state_dict))
    except RuntimeError:
        warnings.warn("loading pretrained model with strict=False")
        model.load_state_dict(remap_legacy_state_dict_keys(state_dict), strict=False)
    print(f"loaded pretrained model from {checkpoint_path}")


def blank_p(logits, tokenizer):
    lset = logits.detach().cpu()
    if torch.rand(1) < 0.05: # print 5 percent of the time
        print(tokenizer.decode([el for el in lset[0].argmax(dim=-1).tolist() if el != lset.shape[-1]-1]))
    lset = rearrange(lset, 'b n v -> (b n) v')
    lset_max = lset.argmax(dim=-1)
    lset_max = lset_max[lset_max == (lset.shape[-1]-1)]
    blank_p = lset_max.shape[0] / lset.shape[0]
    return blank_p


def replace_with_unk_fn(zipf, tokenizer): # zipf: pd.DataFrame
    import string, pandas as pd
    table = str.maketrans('', '', string.punctuation + string.digits + string.whitespace)
    strip_clean = lambda s: s.translate(table).lower()
    zipf = pd.concat([zipf, pd.DataFrame({'Word': ['i'], 'Zipf-value': [7]})], ignore_index=True)
    def replace_with_unk(s, unk_id=1):
        words = s.split()
        result = []
        for i, word in enumerate(words):
            clean_word = strip_clean(word)
            match = zipf.loc[zipf['Word'] == clean_word]
            if not match.empty:
                if list(match['Zipf-value'])[0] >= 4:
                    result.extend(tokenizer.encode(word))
                else:
                    result.append(unk_id)
            else:
                result.append(unk_id)
        return result
    return replace_with_unk

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

def prepare_prompt(
        unpadded:bool,
        txt:List[int], 
        other_args:Dict[str, Any],
        device:torch.device,
        prev_text_outputs:List[str]=None,
        first_pass_text_outputs:List[str]=None, 
        cur_selection_mask:torch.Tensor=None, 
        pad_id=0, 
        bos_id=0,
        loss_on_previous:bool = False,
    ):
    if unpadded: 
        # if we are using CTC history then we don't pad text when preparing data, this is because we only have the previous output once its processed
        # therefore here we need to pad the current text for the encoder, and prepare and pad the prompt + current text for the decoder!
        padded_txt = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(el) for el in txt], batch_first=True, padding_value=pad_id)
    else:
        assert isinstance(txt, torch.Tensor), 'txt should be a (padded) tensor if not using ctc history'
        padded_txt = txt 

    assert prev_text_outputs == None or first_pass_text_outputs == None, 'conditioning on both not implemented yet'

    if prev_text_outputs == None and first_pass_text_outputs == None: return None, padded_txt, other_args
    if cur_selection_mask != None and prev_text_outputs != None: 
        prev_text_outputs = [el for i, el in enumerate(prev_text_outputs) if cur_selection_mask[i]]
    
    assert unpadded, 'something has gone wrong!'

    lm_text_sequence = []
    lm_text_sequence_lengths = []
    prev_lengths = []

    if first_pass_text_outputs != None:
        assert len(first_pass_text_outputs) == len(txt)
        for i, fpass in enumerate(first_pass_text_outputs):
            cur_txt = txt[i]
            full_txt = fpass + [bos_id] + cur_txt
    
            lm_text_sequence.append(torch.LongTensor(full_txt))
            lm_text_sequence_lengths.append(len(full_txt))
            prev_lengths.append(len(fpass))
    else:
        assert len(prev_text_outputs) == len(txt)
        for i, prev_txt in enumerate(prev_text_outputs):
            cur_txt = txt[i]
            full_txt = prev_txt + [bos_id] + cur_txt
            lm_text_sequence.append(torch.LongTensor(full_txt))
            lm_text_sequence_lengths.append(len(full_txt))
            prev_lengths.append(len(prev_txt))

    lm_text_sequence_lengths = torch.LongTensor(lm_text_sequence_lengths)
    lm_text_sequence = torch.nn.utils.rnn.pad_sequence(lm_text_sequence, batch_first=True, padding_value=pad_id)

    if loss_on_previous == False:
        prev_lengths = torch.LongTensor(prev_lengths)
        lm_loss_mask = torch.arange(lm_text_sequence.shape[1]).expand(len(prev_lengths), lm_text_sequence.shape[1]) < prev_lengths.unsqueeze(1)
        other_args['lm_loss_mask'] = lm_loss_mask.to(device)

    other_args['lm_text_sequence'] = lm_text_sequence.to(device)
    other_args['lm_text_sequence_lengths'] = lm_text_sequence_lengths.to(device)
    return prev_text_outputs, padded_txt, other_args
    


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
    scaler = GradScaler() 
    clip_value = args.config['training'].get('clip_value', 0.8) 
    random.seed(args.config['training'].get('random_seed', 12345))
    wandb_config = args.config['wandb']
    dtype = get_dtype(args.config['training'].get('dtype', 'bfloat16'))
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    tokenizer = dataloader.tokenizer

    bos_id = model.get_bos_id()
    pad_id = model.get_pad_id()
    prev_id = model.get_prev_id()
    blank_id = model.get_blank_id()
    first_pass_id = model.get_first_pass_id()

    condition_on_previous = args.config['training'].get('condition_on_previous', False)
    loss_on_previous = args.config['training'].get('loss_on_previous', False)
    use_ctc_history = args.config['training'].get('use_ctc_history', False)
    condition_on_ctc_first_pass = args.config['training'].get('condition_on_ctc_first_pass', False)

    first_pass_forcing_percentage = args.config['training'].get('first_pass_forcing_percentage', 0.0)

    prepad_text = use_ctc_history == False and condition_on_ctc_first_pass == False


    masked_first_pass = args.config['training'].get('masked_first_pass', False)
    if masked_first_pass:
        zipf_freq_path = args.config['data'].get('zipf_freq_path', None)
        assert zipf_freq_path != None, 'must provide zipf_freq_path if masked_first_pass is True'
        import pandas as pd
        zipf_freq = pd.read_csv(zipf_freq_path)
        replace_with_unk = replace_with_unk_fn(zipf_freq, tokenizer)



    if condition_on_ctc_first_pass: assert condition_on_previous == False, 'not implemented yet!'
    if loss_on_previous == True: assert condition_on_previous == True, 'loss_on_previous can only be true if condition_on_previous is true'


    model.train()

    model_dtype = next(model.parameters()).dtype
    

    backprop_every, backwards_every = args.config['training']['backprop_every'], args.config['training'].get('backwards_every', 1)
    assert backprop_every >= backwards_every, f'backprop_every ({backprop_every}) must be >= backwards_every ({backwards_every})'
    
    batch_size = args.config['training']['batch_size']

    
    wandb_loss_accum = {}

    chunk_size, chunk_overlap = args.config.audio_chunking['size'], 0 # previously args.config.audio_chunking['overlap'] though this is not used anymore

    if exists(sequence_scheduler):
        chunk_size = sequence_scheduler.cur_sequence_length
        batch_size = sequence_scheduler.cur_batch_size

    last_podcast, cur_podcast, podcasts_since_last_save = step, step, 0
    max_epochs = args.config['training'].get('max_epochs', 1)

    i, finished = -1, False
    dataloader_iter = iter(dataloader)
    total_recordings = dataloader.total_recordings() * max_epochs
    pbar = tqdm(total = len(dataloader), desc = f'Training - Epoch {epoch}')
    start_spec_augment_after_n_epochs = args.config['training'].get('start_spec_augment_after_n_epochs', -1)

    counter = KeepCount()

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

        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        txt_chunks = [chunk_text_json(text = el, chunk_size = chunk_size, chunk_overlap = chunk_overlap, spectogram_length = audio.shape[-1]) for el in txt] # becomes v slow for v large batch sizes !!

        del audio
        backwards_every_loss = 0.0
        chunks, culm_lengths_audio, nans_in_a_row = [], torch.zeros_like(audio_lengths), 0


        ################################
        for ix, el in enumerate(audio_chunks_):

            remove_mask = ~(culm_lengths_audio > audio_lengths)
            cur_chunks, cur_culm_lengths = el[remove_mask], culm_lengths_audio[remove_mask]
            cur_lengths = cur_chunks.shape[-1] - (cur_culm_lengths + cur_chunks.shape[-1] - audio_lengths[remove_mask] - chunk_overlap).clamp(0)
          
            enc_txt_chunks = [tokenizer.encode(el[ix]) for i, el in enumerate(txt_chunks) if remove_mask[i]]
            enc_txt_chunks_lengths = torch.LongTensor([len(el) for el in enc_txt_chunks])

            if prepad_text == True:
                enc_txt_chunks = [torch.LongTensor(el) for el in enc_txt_chunks]
                enc_txt_chunks = torch.nn.utils.rnn.pad_sequence(enc_txt_chunks, batch_first=True, padding_value=pad_id)

            if condition_on_previous == True and prepad_text == True:
                lm_txt_chunks = []
                prev_lengths = []
                for i, tx_el in enumerate(txt_chunks):
                    if remove_mask[i]: 
                        if ix == 0:
                            lm_txt_chunks.append(torch.LongTensor([bos_id] + tokenizer.encode(tx_el[ix])))
                            prev_lengths.append(0)
                        else:
                            prev = [prev_id] + tokenizer.encode(tx_el[ix-1]) + [bos_id]
                            lm_txt_chunks.append(torch.LongTensor(prev + tokenizer.encode(tx_el[ix])))
                            prev_lengths.append(len(prev) - 1) # - 1 for bos_id

                lm_txt_chunks_lengths = torch.LongTensor([el.shape[0] for el in lm_txt_chunks])
                lm_txt_chunks = torch.nn.utils.rnn.pad_sequence(lm_txt_chunks, batch_first=True, padding_value=pad_id)
                lm_loss_mask = None
                if loss_on_previous == False:
                    prev_lengths = torch.LongTensor(prev_lengths)
                    lm_loss_mask = torch.arange(lm_txt_chunks.shape[1]).expand(len(prev_lengths), lm_txt_chunks.shape[1]) < prev_lengths.unsqueeze(1)
            elif masked_first_pass and prepad_text == True:
                lm_txt_chunks = []
                for i, tx_el in enumerate(txt_chunks):
                    if remove_mask[i]:
                        redacted_text = replace_with_unk(tx_el[ix])
                        lm_txt_chunks.append(torch.LongTensor([bos_id] + redacted_text + [bos_id] + tokenizer.encode(tx_el[ix])))
                lm_txt_chunks_lengths = torch.LongTensor([el.shape[0] for el in lm_txt_chunks])
                lm_txt_chunks = torch.nn.utils.rnn.pad_sequence(lm_txt_chunks, batch_first=True, padding_value=pad_id)
                lm_loss_mask = None
            else:
                lm_txt_chunks = None
                lm_txt_chunks_lengths = None
                lm_loss_mask = None

            if enc_txt_chunks_lengths.max() == 0:
                continue # skip if none contain text (bad batch)
            chunks.append({
                'audio':cur_chunks,
                'txt':enc_txt_chunks,
                'txt_lengths':enc_txt_chunks_lengths,
                'audio_lengths':cur_lengths,
                'selection_mask':remove_mask,
                'cur_culm_lengths':cur_culm_lengths,
                'lm_txt':lm_txt_chunks,
                'lm_txt_lengths':lm_txt_chunks_lengths,
                'lm_loss_mask':lm_loss_mask,
                'ix':ix,
            })
            culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (chunk_overlap if ix != 0 else 0)

        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                scheduler.set_cosine_schedule(total_recordings=total_recordings, cur_podcast=cur_podcast)
        prev_selection_mask, last_kv_set = None, None # selection mask from previous chunk
        prev_text_outputs = None
        ################################
        # shuffle chunks
        if not use_ctc_history: # if using real history we cant shuffle as needs to processed in real order
            chunks = random.sample(chunks, len(chunks))
        

        try:
            for ix, chunk_json in enumerate(chunks):
                print(f'chunk {ix}/{len(chunks)}')
               
                audio, a_lengths = chunk_json['audio'], chunk_json['audio_lengths']
                txt, t_lengths = chunk_json['txt'], chunk_json['txt_lengths']
                lm_txt, lm_t_lengths = chunk_json['lm_txt'], chunk_json['lm_txt_lengths']
                #print(chunk_json['ix'], '----')
                
                other_args = {}
                if lm_txt != None:
                    other_args['lm_text_sequence'] = lm_txt.to(device)
                    other_args['lm_text_sequence_lengths'] = lm_t_lengths.to(device)
                    if chunk_json['lm_loss_mask'] != None: other_args['lm_loss_mask'] = chunk_json['lm_loss_mask'].to(device)

                
                selection_mask = chunk_json['selection_mask']

                cur_selection_mask = None
                if prev_selection_mask != None and not torch.allclose(selection_mask, prev_selection_mask): cur_selection_mask = selection_mask[prev_selection_mask]
                
                with autocast(device.type, dtype=dtype) if torch.cuda.is_available() else nullcontext():
                    audio, a_lengths = audio.to(device, dtype=model_dtype), a_lengths.to(device)
                    audio = apply_augmentation(audio=audio, lengths=a_lengths, augmentation=augmentation, start_augment_after_n_epochs=start_spec_augment_after_n_epochs, epoch=epoch, is_warmup=scheduler.is_warmup)
                    cached_kvs = last_kv_set.clone() if last_kv_set != None else None
                    cached_kv_lengths = torch.LongTensor([cached_kvs.shape[1]] * cached_kvs.shape[0]).to(device) if cached_kvs != None else None

                    if cur_selection_mask != None and cached_kvs != None:
                        cached_kvs = cached_kvs[cur_selection_mask]
                        cached_kv_lengths = cached_kv_lengths[cur_selection_mask]

                    first_pass_text_outputs = None

                    if condition_on_ctc_first_pass:
                        encoder_out = model.forward(audio_signal=audio, length=a_lengths)
                        ctc_output = encoder_out['final_posteriors_ctc']
                        ctc_text = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=blank_id)(ctc_output, decode=False)
                        first_pass_text_outputs = [[first_pass_id] + el for el in ctc_text]
     
                        if first_pass_forcing_percentage > 0.0:
                            first_pass_text_outputs_ = []
                            for i, el in enumerate(first_pass_text_outputs):
                                if random.random() * 100 < first_pass_forcing_percentage:
                                    first_pass_text_outputs_.append([first_pass_id] + txt[i])
                                else:
                                    first_pass_text_outputs_.append(el)
                            first_pass_text_outputs = first_pass_text_outputs_


                        other_args['encoder_outputs'] = encoder_out # avoid recalculating encoder outputs 

                    prev_text_outputs, txt, other_args = prepare_prompt(
                        unpadded=not prepad_text, 
                        prev_text_outputs=prev_text_outputs, 
                        first_pass_text_outputs=first_pass_text_outputs,
                        cur_selection_mask=cur_selection_mask, 
                        txt=txt, 
                        other_args=other_args,
                        device=device,
                        loss_on_previous=loss_on_previous,
                        bos_id=bos_id,
                        pad_id=pad_id,
                    )

                    if 'prefix_to_generate' in args.config['training']: other_args['prefix_to_generate'] = args.config['training']['prefix_to_generate']                    

                    out = model.calc_loss(
                        audio_signal = audio, 
                        text_sequence = txt.to(device),
                        a_lengths = a_lengths,
                        t_lengths = t_lengths.to(device),
                        tokenizer = tokenizer,
                        **other_args,
                    )
                    
                    cur_probs = out.get('ctc_posteriors', None)
                    loss = out['loss']

                    if use_ctc_history:
                        assert cur_probs != None, 'cur_probs must be returned if using ctc history'
                        prev_text_outputs = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=blank_id)(cur_probs, decode=False)
                        prev_text_outputs = [[prev_id] + el for el in prev_text_outputs]
                    
                    
                blank_prob = blank_p(cur_probs.detach(), dataloader.tokenizer) if exists(cur_probs) else None
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

                backwards_every_loss += loss
                counter['steps_since_backwards'] += 1
                counter['steps_since_backprop'] += 1
                
                wandb_loss_accum = {k: wandb_loss_accum.get(k, 0) + v for k, v in out['display_losses'].items()} 

                if (ix+1) % backwards_every == 0 or (ix+1) == len(chunks):
                    scaler.scale(((backwards_every_loss) / (chunk_size*batch_size*counter['steps_since_backwards'])) * 100).backward() # divide by chunk*batch_size constant to weight smaller batches less
                    last_kv_set.detach_() if last_kv_set != None else None
                    counter['steps_since_backwards'] = 0
                    backwards_every_loss = 0

                if (ix+1) % backprop_every == 0 or (ix+1) == len(chunks): 
                    wandb_loss_accum = {k: v / counter['steps_since_backprop'] for k, v in wandb_loss_accum.items()}
                    counter['steps_since_backprop'] = 0
                    full_loss = wandb_loss_accum['loss'] 
           
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
                            **wandb_loss_accum,
                            'blank_p': blank_prob,
                            'learning_rate': learning_rate,
                            'sequence_length': chunk_size,
                            'batch_size': batch_size,
                            'epoch': epoch,
                            'spec_augment': int(True) if start_spec_augment_after_n_epochs != -1 and epoch >= start_spec_augment_after_n_epochs and scheduler.is_warmup == False else int(False),
                        })
                    
                    wandb_loss_accum = {}
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
                dataloader.update(
                    batch_size = batch_size,
                    seen_ids = seen_ids,
                )
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
        wandb_dir = args.config['wandb'].get('dir', './wandb')
        config = OmegaConf.to_container(args.config, resolve=True)
        wandb.init(project=project_name, config=config, name=run_name, dir=wandb_dir) if w_id == '' else wandb.init(project=project_name, id=w_id, resume="must", config=config, allow_val_change=True, dir=wandb_dir)
        wandb.watch(model, log="all") # sometimes this causes a crash ):
        wandb.config.update({'total_params': tparams}, allow_val_change=True)
        print(f'\nLoggging with Wandb id: {wandb.run.id}\n')
        args.config['wandb']['id'] = wandb.run.id # add wandb config to args.config
        if wandb_config.get('update_config_with_wandb_id', False): OmegaConf.save(config=args.config, f=args.config_path)

    model = model.to(device)
    optimizer, scheduler = load_optimizer(args.config, model)

    sequence_scheduler = None
    if 'sequence_scheduler' in args.config:
        method = args.config['sequence_scheduler'].get('method', 'warmup')
        if method == 'warmup':
            sequence_scheduler = SequenceWarmupManager(
                initial_batch_size = args.config['training']['batch_size'],
                initial_sequence_length = args.config['audio_chunking']['size'],
                **args.config['sequence_scheduler']
            )
        elif method == 'random':
            sequence_scheduler = RandomSequenceLengthManager(
                initial_batch_size = args.config['training']['batch_size'],
                initial_sequence_length = args.config['audio_chunking']['size'],
                **args.config['sequence_scheduler']
            )
        else:
            raise ValueError(f'unknown sequence scheduler method: {method}')
        

    pretrained_path = args.config["checkpointing"].get("pretrained", None)
    output_checkpoint_dir = args.config["checkpointing"]["dir"]
    should_initialize_from_pretrained = (
        pretrained_path is not None
        and (args.reset_step or find_latest_checkpoint(output_checkpoint_dir) is None)
    )

    if should_initialize_from_pretrained:
        load_pretrained_model_state(model=model, path=pretrained_path, device=device)
        seen_ids, step, epoch = [], 0, 0
    else:
        seen_ids, step, epoch = load_checkpoint(
            args = args,
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
            sequence_scheduler = sequence_scheduler,
            path = output_checkpoint_dir,
            device = device
        )
        if args.reset_step:
            seen_ids, step, epoch = [], 0, 0

    print(f'Starting from podcast: {len(seen_ids)}')
    random_seed = args.config['training'].get('random_seed', 1234)
    start_spec_augment_after_n_epochs = args.config['training'].get('start_spec_augment_after_n_epochs', -1)

    # skip data up to step
    dataloader = VariableBatchSimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.config['training']['batch_size'],
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
    
    if sequence_scheduler and dataloader.batch_size != sequence_scheduler.cur_batch_size:
        print('WARNING: dataloader batch size does not match sequence scheduler batch size, updating dataloader batch size')
        dataloader.update(batch_size = sequence_scheduler.cur_batch_size, seen_ids = seen_ids)

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
      
