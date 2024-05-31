import torch
from typing import Dict, List, Tuple

from lcasr.models.sconformer_xl import SCConformerXL
from lcasr.models.mamba import Mamba
from lcasr.models.enc_dec_sconformer import EncDecSconformer
from lcasr.models.enc_dec_sconformer_v2 import EncDecSconformerV2
from lcasr.models.sconformer_meta import SCConformerMeta
from lcasr.models.hourglassconformer import HourGlassConformer
# from lcasr.models.metaconformer import MetaConformer
# from lcasr.models.stconformer import STConformer
from lcasr.utils.scheduling import SequenceWarmupManager, CosineLRScheduler
import os
from tqdm import tqdm

from apex.optimizers import FusedAdam
from torch.optim import Adam
from lcasr.optim import madgrad
import argparse
import warnings

def get_model_class(config:Dict={}, args:argparse.Namespace={}):
    model_classes = [
        'SCConformerXL', 
        'Mamba', 
        'EncDecSconformer', 
        'EncDecSconformerV2',
        'SCConformerMeta',
        'HourGlassConformer',
    ]
    
    if 'model_class' in args:
        model_class = args.model_class
    elif 'model_class' in config:
        model_class = config['model_class']
    else:
        warnings.warn('No model_class specified in model config or args, defaulting to SCConformerXL') 
        model_class = 'SCConformerXL'
    assert model_class in model_classes, f'Unknown model class {model_class}, must be one of {model_classes}'

    if model_class == 'SCConformerXL':
        return SCConformerXL
    elif model_class == 'Mamba':
        return Mamba
    elif model_class == 'EncDecSconformer':
        return EncDecSconformer
    elif model_class == 'EncDecSconformerV2':
        return EncDecSconformerV2
    elif model_class == 'SCConformerMeta':
        return SCConformerMeta
    elif model_class == 'HourGlassConformer':
        return HourGlassConformer
    else:
        raise NotImplementedError(f'Unknown model class {model_class}, must be one of {model_classes}')
    


def load_model(config:Dict, vocab_size, model_class=SCConformerXL):
    model = model_class(**config.model, vocab_size=vocab_size)
    return model

def load_optimizer(config:Dict, model:torch.nn.Module):
    model_device = next(model.parameters()).device.type # check device of model

    optim_type = config['optimizer']['name']
    allowed_types = ['adam', 'madgrad', 'mirrormadgrad']
    
    assert optim_type in allowed_types, f'Unknown optimizer {optim_type}, must be one of {allowed_types}'
    assert model_device in ['cpu', 'cuda'], f'Unknown device {model_device}, must be one of [cpu, cuda]'

    optim_args = config['optimizer']['args']

    if config['optimizer'].get('weight_decay_groups', 'default') == 'default':
        param_groups = model.get_param_groups(optim_args) if hasattr(model, 'get_param_groups') else model.parameters()
    elif config['optimizer'].get('weight_decay_groups', 'default') == 'none':
        param_groups = model.parameters()
    else:
        raise NotImplementedError(f'Unknown weight_decay_groups {config["optimizer"]["weight_decay_groups"]}, must be one of [default, none]')  

    if optim_type == 'adam':
        optimizer = Adam(param_groups, **optim_args) if model_device == 'cpu' else FusedAdam(model.parameters(), **optim_args)
    elif optim_type == 'madgrad':
        optimizer = madgrad.MADGRAD(param_groups, **optim_args) # best
    elif optim_type == 'mirrormadgrad':
        optimizer = madgrad.MirrorMADGRAD(param_groups, **optim_args)

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
        other:Dict={},
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
        **other,
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
        device='cpu',
        other:List[Tuple]=[], # list of tuples (obj, checkpoint key)
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
        warnings.warn('loading model with strict=False')
        model.load_state_dict(checkpoint['model'], strict=False)
        warnings.warn('SETTING OPTIMIZER TO NONE DUE TO NON-STRICT LOAD')
        optimizer = None
    print(f'loaded model from {path}')
    if optimizer != None and 'optimizer' in checkpoint and checkpoint['optimizer'] != None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler != None and 'scheduler' in checkpoint and checkpoint['scheduler'] != None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if sequence_scheduler != None and 'sequence_scheduler' in checkpoint and checkpoint['sequence_scheduler'] != None:
        sequence_scheduler.load_state_dict(checkpoint['sequence_scheduler'])
  
    for obj, key in other:
        if key in checkpoint:
            obj.load_state_dict(checkpoint[key])
        else:
            warnings.warn(f'Could not find {key} in checkpoint, skipping')

    seen_ids, epoch, step = checkpoint.get('seen_ids', []), checkpoint.get('epoch', 0), checkpoint.get('podcast_step', 0)
    return seen_ids, step, epoch


def avg_all_models_in_dir(path:str, out_path:str, model_name:str='step_105360.pt'):
    all_folders = os.listdir(path)
    all_folders = [el for el in all_folders if os.path.exists(os.path.join(path, el, model_name))]
    total_models = len(all_folders)
    avg_model = None
    model_checkpoint = None
    for folder in tqdm(all_folders):
        model_path = os.path.join(path, folder, model_name)
  
        model = torch.load(model_path, map_location='cpu')
        state_dict = model['model']
        if avg_model is None:
            avg_model = {key:state_dict[key] * (1/total_models) for key in state_dict.keys()}
            model_checkpoint = {key:model[key] for key in model.keys() if key not in ['model', 'optimizer', 'scheduler']}
        else:
            for key in state_dict.keys():
                avg_model[key] += state_dict[key] * (1/total_models)
    
    model_checkpoint['model'] = avg_model
    torch.save(model_checkpoint, out_path)

class KeepCount:
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            setattr(self, key, 0)
            return getattr(self, key)
        
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)
    
    def reset(self):
        self.__dict__ = {}

class argsclass:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return str(self.__dict__)
    
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return None
        
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)
    
    def __iter__(self):
        return self.__dict__.__iter__()
    
    def __len__(self):
        return len(self.__dict__)