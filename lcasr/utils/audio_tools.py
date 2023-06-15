import torch, torchaudio
from typing import Tuple, Union, List, Optional, Dict
import math
import os
from tqdm import tqdm

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

def to_spectogram(waveform:torch.Tensor):
    '''some of config i,e win and n_fft is from: https://github.com/robflynnyh/NeMo/blob/main/nemo/collections/asr/parts/preprocessing/features.py
    '''
    return torchaudio.transforms.MelSpectrogram(
        win_length = 320,
        hop_length = 160,
        n_fft = 2 ** math.ceil(math.log2(320)), # 512
        n_mels = 64,
        normalized = 'window'
    )(waveform)

def total_seconds(spectogram_length:int) -> float:
    return (spectogram_length * 160) / 16000

def processing_chain(path_in:str):
    waveform, sample_rate = load(path_in)
    waveform = grab_left_channel(waveform)
    waveform = resample(waveform, sample_rate, 16000)
    spectrogram = to_spectogram(waveform)
    return spectrogram


def chunk_spectogram(
        spec: torch.Tensor, # mel spectrogram (batch, features, time)
        chunk_size: int,
        chunk_overlap: int,
    ):
    assert len(spec.shape) == 3, "Audio must be 3D i.e. (batch, features, time)"
    assert chunk_size > chunk_overlap, "chunk_size must be greater than chunk_overlap"
    
    splits = []
    for i in range(0, spec.shape[2], chunk_size - chunk_overlap):
        splits.append(spec[:, :, i:i+chunk_size])
    return splits

def findall_files_spotify(path:str, ext:str, verbose = True) -> List[str]:
    '''
    goes to root of the directory for every folder and finds all files with given extension
    '''
    audio_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                pth = os.path.join(root, file)
                #print(f' adding {pth}') if verbose else None
                print(pth) if len(pth) != 107 and verbose else None
                audio_files.append(pth)
    return audio_files


def pair_audio_txt(audio_path:str, txt_path:str, txt_ext:str, audio_ext, verbose=True) -> Dict[str, Dict[str, str]]:
    audio_f = findall_files_spotify(audio_path, audio_ext)
    txt_f = findall_files_spotify(txt_path, txt_ext, verbose=False)
    pair_f = {}

    for audio_p in (tqdm(audio_f) if verbose else audio_f):
        ref_path = "_".join([el.split(' ')[0] for el in audio_p.split('/')[-4:]]).replace(audio_ext, '')
        pair_f[ref_path] = {'audio': audio_p}
    for txt_p in (tqdm(txt_f) if verbose else txt_f):
        ref_path = "_".join(txt_p.split('/')[-4:]).replace(txt_ext, '')
        pair_f[ref_path]['txt'] = txt_p

    return pair_f


