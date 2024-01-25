import json
from typing import Dict
import torch
from omegaconf.omegaconf import OmegaConf

def exists(item):
    return item is not None

def load_json(jfile:str) -> Dict:
    with open(jfile, 'r') as f:
        return json.load(f)

def load_pairs(pairs:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json') -> Dict:
    return load_json(pairs)

def get_config_from_checkpoint(checkpoint_path:str, out_path:str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    OmegaConf.save(config, out_path)