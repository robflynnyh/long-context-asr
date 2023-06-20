import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.audio_tools import SimpleDataloader
from lcasr.utils.audio_tools import chunk_spectogram

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
        scheduler:torch.optim.lr_scheduler._LRScheduler = None,
    ):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

def train(
        args:argparse.Namespace,
        model:torch.nn.Module, 
        dataloader:torch.utils.data.DataLoader, 
        optimizer:torch.optim.Optimizer, 
        device:torch.device
    ):
    model.train()
    model_dtype = next(model.parameters()).dtype
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='mean')

    cur_tokens_in_loss = 0
    cur_loss = torch.tensor(0.0, dtype=model_dtype, device=device)

    for batch in tqdm(dataloader):
        chunks = batch['chunks']
        
        last_prob_set = None # last set of probabilities output by model

        for ix, chunk_json in enumerate(chunks):
            print(f'chunk {ix}/{len(chunks)}')
            audio, a_lengths = chunk_json['audio'], chunk_json['audio_lengths']
            txt, t_lengths = chunk_json['txt'], chunk_json['txt_lengths']
            print(audio.shape, txt.shape)
            audio = audio.to(device, dtype=model_dtype)
            out = model(audio_signal = audio, length = a_lengths)
            print(a_lengths)
            loss = ctc_loss_fn(out['final_posteriors'].transpose(0,1), txt, out['length'], t_lengths)
            backwards_pass(loss, optimizer)
            print(loss.item())
            print(out.keys())
            
        # txt = txt.to(device)
        # optimizer.zero_grad()
        # loss = model(audio, txt)
        # loss.backward()
        # optimizer.step()
        # print(loss.item())

def main(args):
    args.config = OmegaConf.load(args.config)
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    paired_data = lcasr.utils.audio_tools.load_json('/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json')
    print(args.config.audio_chunking)
    dataloader = SimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = args.batch_size,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.backends.cuda.enable_flash_sdp(enabled=True) # enable flash attention if cuda for faster training

    model = model.to(device)
    optimizer = load_optimizer(args.config, model)
    train(args, model, dataloader, optimizer, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-b', '--batch_size', type=int, default=3, help='batch size')

    args = parser.parse_args()
    main(args)
    