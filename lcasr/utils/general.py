import torch
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
import os

def load_model(config:Dict, vocab_size):
    model = SCConformerXL(**config.model, vocab_size=vocab_size)
    return model

def save_model(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler,
        podcast_step:int,
        config:Dict,
    ):
    save_path = os.path.join(config['checkpointing']['dir'], f'step_{podcast_step}.pt')
    save_dict = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict() if scheduler is not None else None,
        'podcast_step':podcast_step,
        'config':config,
    }
    torch.save(save_dict, save_path)

def find_latest_checkpoint(path:str = './checkpoints'):
    checkpoints = [el for el in os.listdir(path) if el.endswith('.pt')]
    if len(checkpoints) == 0:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return checkpoints[-1]


def load_checkpoint(model, optimizer=None, path='./checkpoints'):
    latest_checkpoint = find_latest_checkpoint(path)
    if latest_checkpoint is None:
        return 0
    path = os.path.join(path, latest_checkpoint)
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        print('loading model with strict=False')
        model.load_state_dict(checkpoint['model'], strict=False)
        print('SETTING OPTIMIZER TO NONE DUE TO NON-STRICT LOAD')
        optimizer = None
    print(f'loaded model from {path}')
    if optimizer != None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
  
    step = checkpoint['podcast_step']
    return step