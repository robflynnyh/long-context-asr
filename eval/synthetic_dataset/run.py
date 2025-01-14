import torch, lcasr, os, re
import argparse
from tqdm import tqdm
from typing import List, Tuple
from lcasr.utils.audio_tools import to_spectogram
from lcasr.utils.general import load_model, get_model_class
from lcasr.eval.utils import zero_out_spectogram, fetch_logits, decode_beams_lm
from lcasr.eval.wer import word_error_rate_detail 
from pyctcdecode import build_ctcdecoder
import time
from functools import partial
import pickle as pkl
import torchaudio

defaults = {
    'earnings22': "/mnt/parscratch/users/acp21rjf/synthetic_earnings22_test/",
    'rev16': "/mnt/parscratch/users/acp21rjf/synthetic_rev16_test/",
    'this_american_life': '/mnt/parscratch/users/acp21rjf/synthetic_TAL_test/',
}

from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()


def fetch_data(path:str):
    files = os.listdir(path)
    files = sorted(files)
    full_path_files = [os.path.join(path, el) for el in files]
    return full_path_files, files


def process_text_and_audio_fn(rec_dict):
    path = rec_dict['file']
    with open(path, 'rb') as f:
        data = pkl.load(f)
    
    all_text = " ".join([el.text.strip() for el in data['segments']])
    audio = data['synthetic_wavs']
    audio = [audio[el] for el in range(len(audio))]
    audio = torch.cat(audio, dim=-1)

    # torchaudio.save("test.wav", audio, 16000)
    # print(all_text)
    
    audio_spec = to_spectogram(audio, global_normalisation=True)
    return audio_spec, normalize(all_text).lower().strip()




def get_text_and_audio(split='test', dataset='earnings22', fullpath=None):
    assert split == 'test', 'Only test split is supported'
    data_path = defaults[dataset] if fullpath is None else fullpath
    
    full_paths, ids = fetch_data(path=data_path)
    return_data = []
    for rec in range(len(ids)):
        return_data.append({
            'id': ids[rec],
            'file': full_paths[rec],
            "process_fn": process_text_and_audio_fn
        })

    return return_data


if __name__ == "__main__":
    data = get_text_and_audio(dataset='earnings22')
    data[0]['process_fn'](data[0])