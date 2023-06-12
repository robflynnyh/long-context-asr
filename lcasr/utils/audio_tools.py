import torch, torchaudio
from typing import Tuple, Union, List, Optional
import math

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





