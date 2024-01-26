import torch, torchaudio
from typing import Tuple, Union, List, Optional, Dict
import math
import os
from tqdm import tqdm
import json
import sentencepiece as spm
from einops import rearrange
import subprocess
import numpy as np
import librosa
from lcasr.utils.helpers import load_json, exists, load_pairs

WIN_LENGTH = 400
HOP_LENGTH = 160
SR = 16000

def load(path:str) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate

def resample(waveform:torch.Tensor, orig_sr:int, new_sr:int) -> torch.Tensor:
    return torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)(waveform)

def save(waveform:torch.Tensor, sample_rate:int, path:str) -> None:
    torchaudio.save(path, waveform, sample_rate)

def grab_left_channel(waveform:torch.Tensor) -> torch.Tensor:
    if len(waveform.shape) == 2:
        return waveform[0, None]
    elif len(waveform.shape) == 1:
        return waveform[None]
    else:
        raise ValueError("Waveform must be 1D or 2D")

def take_mean_channel(waveform:torch.Tensor) -> torch.Tensor:
    if len(waveform.shape) == 2:
        return waveform.mean(0, keepdim=True)
    elif len(waveform.shape) == 1:
        return waveform[None]
    else:
        raise ValueError("Waveform must be 1D or 2D")

def to_spectogram(waveform:torch.Tensor, global_normalisation=True):
    '''some of config i,e win and n_fft is from: https://github.com/robflynnyh/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py
    '''
    spec = torchaudio.transforms.MelSpectrogram(
        win_length = WIN_LENGTH,
        hop_length = HOP_LENGTH,
        n_fft = 2 ** math.ceil(math.log2(WIN_LENGTH)), # 512
        n_mels = 80,
        normalized = False
    )(waveform)
    # normalize
    if global_normalisation:
        spec = (spec - spec.mean(-1, keepdim=True)) / spec.std(-1, keepdim=True)
    return spec

def total_seconds(spectogram_length:int) -> float:
    '''converts number of frames to seconds'''
    return (spectogram_length * HOP_LENGTH) / SR

def total_frames(seconds:float) -> int:
    '''inverse of total_seconds'''
    return int((seconds * 16000) / HOP_LENGTH) 

def processing_chain(path_in:str, normalise:bool = True):
    waveform, sample_rate = load(path_in)
    waveform = grab_left_channel(waveform)
    waveform = resample(waveform, sample_rate, SR)
    spectrogram = to_spectogram(waveform, global_normalisation=normalise)
    return spectrogram

        
def delete_all_spectograms(pairs:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json'):
    pairs = load_json(pairs)
    sure = input(f'Are you sure you want to delete {len(pairs)} spectograms? (y/n)')
    if sure == 'y':
        for k in tqdm(list(pairs.keys())):
            if os.path.exists(pairs[k]['audio']):
                os.remove(pairs[k]['audio'])
    else:
        print('Aborted')




def findall_files_spotify(path:str, ext:str, verbose = True) -> List[str]:
    '''
    goes to root of the directory for every folder and finds all files with given extension
    '''
    audio_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                pth = os.path.join(root, file)
                print(f' adding {pth}') if verbose else None
                audio_files.append(pth)
    return audio_files

# audio /mnt/parscratch/users/acp21rjf/spotify/audio/ .spec.pt
# txt /mnt/parscratch/users/acp21rjf/spotify/txt/spotify-podcasts-2020/podcasts-transcripts/ .json

def pair_audio_txt(
        audio_path:str = '/mnt/parscratch/users/acp21rjf/spotify/audio/',
        txt_path:str = '/mnt/parscratch/users/acp21rjf/spotify/txt/spotify-podcasts-2020/podcasts-transcripts/',
        txt_ext:str = '.json',
        audio_ext = '.spec.pt',
        save_path:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json',
        verbose=False
    ) -> Dict[str, Dict[str, str]]:

    audio_f = findall_files_spotify(audio_path, audio_ext, verbose=verbose)
    txt_f = findall_files_spotify(txt_path, txt_ext, verbose=verbose)
    pair_f = {}

    for audio_p in (tqdm(audio_f, desc='pairing audio and txt') if verbose else audio_f):
        ref_path = "_".join([el.split(' ')[0] for el in audio_p.split('/')[-4:]]).replace(audio_ext, '')
        pair_f[ref_path] = {'audio': audio_p}
    for txt_p in (tqdm(txt_f) if verbose else txt_f):
        ref_path = "_".join(txt_p.split('/')[-4:]).replace(txt_ext, '')
        pair_f[ref_path]['txt'] = txt_p

    if save_path != '' and save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(pair_f, f)
        print(f'pairs saved to {save_path}')
    return pair_f

def get_audio_duration(audio_path:str) -> float:
    '''returns duration of audio in seconds using ffprobe'''
    command = f'ffprobe -i "{audio_path}" -show_entries format=duration -v quiet -of csv="p=0"'
    duration = subprocess.check_output(command, shell=True)
    return float(duration)

def append_timings_to_json(
        paired_json_path:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json'
    ):
    pairs = load_json(paired_json_path)
    for key in tqdm(list(pairs.keys())):
        audio_ogg_path = pairs[key]['audio'].replace('.spec.pt', '.ogg')
        duration = get_audio_duration(audio_ogg_path)
        pairs[key]['duration'] = duration

    with open(paired_json_path, 'w') as f:
        json.dump(pairs, f)
    print(f'pairs saved to {paired_json_path}')

def retrieve_all_text(
        audio_txt_pairs_f:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json',
        save_path:str = '/mnt/parscratch/users/acp21rjf/spotify/all_text.txt',
    ) -> List[str]:
    pairs = load_json(audio_txt_pairs_f)
    all_text = []
    
    to_save = save_path != '' and save_path is not None

    for key in tqdm(list(pairs.keys())): # faster ways to do this 
        cur_json = load_json(pairs[key]['txt'])
        text = " ".join([el['word'] for el in cur_json['results'][-1]['alternatives'][0]['words']])
        all_text.append(text)
        if to_save:
            with open(save_path, 'a') as f:
                f.write(text + '\n')
    return all_text

def train_tokenizer(
        raw_txt:str = '/mnt/parscratch/users/acp21rjf/spotify/all_text.txt',
        save_path:str = '/mnt/parscratch/users/acp21rjf/spotify/',
        vocab_size:int = 4095,
    ):
    spm.SentencePieceTrainer.train(
        input=raw_txt,
        model_prefix='tokenizer',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        max_sentence_length=1000000, #
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=-1,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        normalization_rule_name='nmt_nfkc_cf'
    )
    os.system(f'mv tokenizer.model {save_path}')
    os.system(f'mv tokenizer.vocab {save_path}')

def load_tokenizer(
        tokenizer_path:str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../artifacts/tokenizer.model') # relative path to tokenizer
    ):
    return spm.SentencePieceProcessor(model_file=tokenizer_path)





