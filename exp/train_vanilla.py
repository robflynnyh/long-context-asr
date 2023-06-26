import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.audio_tools import SimpleDataloader
from lcasr.utils.audio_tools import chunk_spectogram

from torch.cuda.amp import GradScaler
from torch import autocast

from apex.optimizers import FusedAdam
from torch.optim import Adam

def load_model(config:Dict, vocab_size):
    model = SCConformerXL(**config.model, vocab_size=vocab_size)
    return model

def load_optimizer(config:Dict, model:torch.nn.Module):
    # check device
    model_device = next(model.parameters()).device.type
    if model_device == 'cpu':
        optimizer = Adam(model.parameters(), **config.optimizer)
    elif model_device == 'cuda':
        print('-- Using FusedAdam --')
        optimizer = FusedAdam(model.parameters(), **config.optimizer)
    else:
        raise ValueError('Unknown device')
    return optimizer

def backwards_pass(
        loss:torch.Tensor,
        optimizer:torch.optim.Optimizer,
        scaler:GradScaler,
        scheduler:torch.optim.lr_scheduler._LRScheduler = None,
    ):
    optimizer.zero_grad()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

def train(
        args:argparse.Namespace,
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer, 
        device:torch.device
    ):
    scaler = GradScaler()

    model.train()
    model_dtype = next(model.parameters()).dtype
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='mean')

    overlap = args.config.audio_chunking['overlap']
    ds_overlap = overlap // 4 # 4 is the subsampling factor
    backprop_every = args.backprop_every


    cur_tokens_in_loss = 0
    cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)

    pbar = tqdm(dataloader)
    for batch in pbar:
        chunks = batch['chunks']
        
        last_prob_set = None # last set of probabilities output by model
        prev_selection_mask = None # selection mask from previous chunk

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
                out = model(audio_signal = audio, length = a_lengths)
                # check for nan
                cur_probs = out['final_posteriors'].clone()
                B,N,C = cur_probs.shape 

                if last_prob_set != None:
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
            
            cur_tokens_in_loss += B * N

            if cur_tokens_in_loss > backprop_every:
                cur_loss /= cur_tokens_in_loss
                print(f'loss: {cur_loss.item()}')
                backwards_pass(cur_loss, optimizer, scaler)
                cur_tokens_in_loss = 0
                last_prob_set.detach_()
                cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)

            prev_selection_mask = selection_mask.clone()



def main(args):
    args.config = OmegaConf.load(args.config)
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    paired_data = lcasr.utils.audio_tools.load_json('/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json')

    dataloader = SimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.batch_size,
        chunk_size = args.config.audio_chunking['size'],
        chunk_overlap = args.config.audio_chunking['overlap'],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = model.to(device)
    optimizer = load_optimizer(args.config, model)
    train(args, model, dataloader, optimizer, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-b', '--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('-bprop', '--backprop_every', type=int, default=20000, help='backprop every n tokens')

    args = parser.parse_args()
    main(args)
    