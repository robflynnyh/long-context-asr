import torch
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from lcasr.utils.scheduling import SequenceWarmupManager, CosineLRScheduler
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


def load_checkpoint(
        args, 
        model, 
        optimizer=None, 
        scheduler:CosineLRScheduler=None,
        sequence_scheduler:SequenceWarmupManager=None,
        path='./checkpoints', 
        device='cpu'
    ):
    latest_checkpoint = find_latest_checkpoint(path)
    if latest_checkpoint is None:
        return 0
    path = os.path.join(path, latest_checkpoint)
    checkpoint = torch.load(path, map_location=device)
    if args and args.remove_scheduler:
        checkpoint['scheduler'] = None
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

    if scheduler != None and 'scheduler' in checkpoint and checkpoint['scheduler'] != None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if sequence_scheduler != None and 'sequence_scheduler' in checkpoint and checkpoint['sequence_scheduler'] != None:
        sequence_scheduler.load_state_dict(checkpoint['sequence_scheduler'])
  
    step = checkpoint['podcast_step'] 
    return step