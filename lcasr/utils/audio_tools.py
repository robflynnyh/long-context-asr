import torch, torchaudio
from typing import Tuple, Union, List, Optional, Dict
import math
import os
from tqdm import tqdm
import json
import sentencepiece as spm
from einops import rearrange

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

def load_json(jfile:str) -> Dict:
    with open(jfile, 'r') as f:
        return json.load(f)

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

def retrieve_all_text(
        audio_txt_pairs_f:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json',
        save_path:str = '/mnt/parscratch/users/acp21rjf/spotify/all_text.txt',
    ) -> List[str]:
    pairs = load_json(audio_txt_pairs_f)
    all_text = []
    
    to_save = save_path != '' and save_path is not None

    for key in tqdm(list(pairs.keys())):
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
        vocab_size:int = 4096,
    ):
    spm.SentencePieceTrainer.train(
        input=raw_txt,
        model_prefix='tokenizer',
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=-1,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
    )
    os.system(f'mv tokenizer.model {save_path}')
    os.system(f'mv tokenizer.vocab {save_path}')

def load_tokenizer(
        tokenizer_path:str = '/mnt/parscratch/users/acp21rjf/spotify/tokenizer.model',
    ):
    return spm.SentencePieceProcessor(model_file=tokenizer_path)


def load_sample(entry:Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
    audio = torch.load(entry['audio'])
    txt = load_json(entry['txt'])
    return audio, txt


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, pairs:Dict[str, Dict[str, str]], tokenizer:spm.SentencePieceProcessor):
        self.pairs = pairs
        self.keys = list(pairs.keys())
        self.tokenizer = tokenizer
        self.bos_id = self.tokenizer.bos_id()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        audio, txt = load_sample(self.pairs[self.keys[idx]])
        txt = " ".join([el['word'] for el in txt['results'][-1]['alternatives'][0]['words']])
        txt = self.tokenizer.encode(txt)
        txt = [self.bos_id] + txt
        txt = torch.LongTensor(txt)

        audio = rearrange(audio, '() f t -> t f')
        return audio, txt


def collate_fn(pad_id=0):
    def collate(batch):
        audio, txt = zip(*batch)
        audio_lengths = torch.LongTensor([el.shape[0] for el in audio])
        txt_lengths = torch.LongTensor([el.shape[0] for el in txt])
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=pad_id)
        txt = torch.nn.utils.rnn.pad_sequence(txt, batch_first=True, padding_value=pad_id)
        audio = rearrange(audio, 'b t f -> b f t')
        return {
            'audio': audio,
            'txt': txt,
            'audio_lengths': audio_lengths,
            'txt_lengths': txt_lengths,
        }
    return collate

class SimpleDataloader(torch.utils.data.DataLoader):
    def __init__(self, pairs:Dict[str, Dict[str, str]], tokenizer:spm.SentencePieceProcessor, batch_size:int = 5):
        self.dataset = SimpleDataset(pairs, tokenizer)
        super().__init__(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn(tokenizer.pad_id()))


