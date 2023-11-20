import torch
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from lcasr.models.metaconformer import MetaConformer
from lcasr.models.stconformer import STConformer
from lcasr.utils.scheduling import SequenceWarmupManager, CosineLRScheduler
import os

from apex.optimizers import FusedAdam
from torch.optim import Adam
import madgrad

def load_model(config:Dict, vocab_size, model_class=MetaConformer):
    model = model_class(**config.model, vocab_size=vocab_size)
    return model

def load_optimizer(config:Dict, model:torch.nn.Module):
    model_device = next(model.parameters()).device.type # check device of model

    optim_type = config['optimizer']['name']
    allowed_types = ['adam', 'madgrad', 'mirrormadgrad']
    
    assert optim_type in allowed_types, f'Unknown optimizer {optim_type}, must be one of {allowed_types}'
    assert model_device in ['cpu', 'cuda'], f'Unknown device {model_device}, must be one of [cpu, cuda]'

    optim_args = config['optimizer']['args']

    if optim_type == 'adam':
        optimizer = Adam(model.parameters(), **optim_args) if model_device == 'cpu' else FusedAdam(model.parameters(), **optim_args)
    elif optim_type == 'madgrad':
        optimizer = madgrad.MADGRAD(model.parameters(), **optim_args) # best
    elif optim_type == 'mirrormadgrad':
        optimizer = madgrad.MirrorMADGRAD(model.parameters(), **optim_args)

    sheduler = CosineLRScheduler(
        optimizer = optimizer,
        warmup_steps = config['scheduler']['warmup_steps'],
        peak_value = config['optimizer']['args']['lr'],
        final_value = 0.0, # decay to 0
    )

    return optimizer, sheduler

def save_model(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler._LRScheduler,
        podcast_step:int,
        config:Dict,
        sequence_scheduler:SequenceWarmupManager=None,
        seen_ids:List[int]=[],
        epoch:int=0,
    ):
    save_path = os.path.join(config['checkpointing']['dir'], f'step_{podcast_step}.pt')
    save_dict = {
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict() if scheduler is not None else None,
        'podcast_step':podcast_step,
        'config':config,
        'sequence_scheduler':sequence_scheduler.state_dict() if sequence_scheduler is not None else None,
        'seen_ids':seen_ids,
        'epoch':epoch,
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
        return [], 0, 0 # seen_ids, step, epoch
    path = os.path.join(path, latest_checkpoint)
    checkpoint = torch.load(path, map_location=device)
    if args and args.remove_scheduler:
        checkpoint['scheduler'] = None
        checkpoint['sequence_scheduler'] = None
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
  
    seen_ids = checkpoint.get('seen_ids', [])
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('podcast_step', 0)
    return seen_ids, step, epoch