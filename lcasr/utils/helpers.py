import json
from typing import Dict

def exists(item):
    return item is not None

def load_json(jfile:str) -> Dict:
    with open(jfile, 'r') as f:
        return json.load(f)

def load_pairs(pairs:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json') -> Dict:
    return load_json(pairs)