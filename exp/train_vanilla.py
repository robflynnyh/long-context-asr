import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf

from lcasr.utils.audio_tools import SimpleDataloader

from apex.optimizers import FusedAdam
from torch.optim import Adam

def load_model(config:str, vocab_size):
    config = OmegaConf.load(config)
    model = SCConformerXL(**config.model, vocab_size=vocab_size)
    return model

def load_optimizer(config:str, model:torch.nn.Module):
    config = OmegaConf.load(config)
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

def train(model:torch.nn.Module, dataloader:SimpleDataloader, optimizer, device):
    model.train()
    model_dtype = next(model.parameters()).dtype
    for batch in tqdm(dataloader):
        audio, txt = batch['audio'], batch['txt']
        audio = audio.to(device, dtype=model_dtype)
        txt = txt.to(device)
        optimizer.zero_grad()
        loss = model(audio, txt)
        loss.backward()
        optimizer.step()
        print(loss.item())

def main(args):
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    paired_data = lcasr.utils.audio_tools.load_json('/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json')
    dataloader = SimpleDataloader(paired_data, tokenizer, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = load_optimizer(args.config, model)
    train(model, dataloader, optimizer, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, required=True, help='path to config file')
    parser.add_argument('-b', '--batch_size', type=int, default=3, help='batch size')

    args = parser.parse_args()
    main(args)
    