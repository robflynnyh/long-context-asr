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
        n_fft = 2 ** math.ceil(math.log2(400)), # 512
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

def processing_chain(path_in:str):
    waveform, sample_rate = load(path_in)
    waveform = grab_left_channel(waveform)
    waveform = resample(waveform, sample_rate, SR)
    spectrogram = to_spectogram(waveform, global_normalisation=True)
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

def chunk_text_json(
        text: List[Dict[str, str]],
        chunk_size: int,
        chunk_overlap: int,
        spectogram_length: int,
        get_seconds: bool = False
    ):
    assert chunk_size > chunk_overlap, "chunk_size must be greater than chunk_overlap"
    
    splits = []
    start_end_times = []
    for i in range(0, spectogram_length, chunk_size - chunk_overlap):
        c_start_pos, c_end_pos = i, i + chunk_size
        c_start_pos_sec, c_end_pos_sec = total_seconds(c_start_pos), total_seconds(c_end_pos)
        c_text = [el['word'] for el in text if float(el['startTime'][:-1]) >= c_start_pos_sec and float(el['endTime'][:-1]) <= c_end_pos_sec]
        splits.append(" ".join(c_text))
        start_end_times.append((c_start_pos_sec, c_end_pos_sec))
    
    return splits if not get_seconds else (splits, start_end_times)
        
def delete_all_spectograms(pairs:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json'):
    pairs = load_json(pairs)
    sure = input(f'Are you sure you want to delete {len(pairs)} spectograms? (y/n)')
    if sure == 'y':
        for k in tqdm(list(pairs.keys())):
            if os.path.exists(pairs[k]['audio']):
                os.remove(pairs[k]['audio'])
    else:
        print('Aborted')


def load_json(jfile:str) -> Dict:
    with open(jfile, 'r') as f:
        return json.load(f)

def load_pairs(pairs:str = '/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json') -> Dict:
    return load_json(pairs)

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
        tokenizer_path:str = '/mnt/parscratch/users/acp21rjf/spotify/tokenizer.model',
    ):
    return spm.SentencePieceProcessor(model_file=tokenizer_path)


def load_sample(entry:Dict[str, str]) -> Tuple[torch.Tensor, torch.Tensor]:
    # audio_name = entry['audio'].replace('.spec.pt', '.ogg')
    # waveform, sample_rate = load(audio_name)
    # waveform = grab_left_channel(waveform)
    # waveform = resample(waveform, sample_rate, SR)
    # waveform = waveform.numpy()
    # spec = librosa.feature.melspectrogram(
    #     y=waveform, 
    #     sr=SR, 
    #     n_fft=512, 
    #     hop_length=HOP_LENGTH, 
    #     n_mels=80, 
    # )
    # audio = torch.as_tensor(spec, dtype=torch.float32)
    # audio = processing_chain(audio_name)
    audio = torch.load(entry['audio'])
    txt = load_json(entry['txt'])
    return audio, txt


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            pairs:Dict[str, Dict[str, str]],
            batch_size:int = 8,
            subgroup_shuffle_size:int = 2000,
            skip_to:int = 0,
            check_exists:bool = False, # if preprocessing hasn't finished, set to True
        ):
        self.batch_size = batch_size
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.pairs = pairs
        self.items = sorted(list(pairs.items()), key=lambda x: x[1]['duration'])
        if check_exists:
            self.remove_nonexistent()
        self.create_batches()
        self.items = self.items[skip_to:]

    def remove_nonexistent(self):
        items = []
        for el in tqdm(self.items, desc='removing nonexistent files'):
            if os.path.exists(el[1]['audio']):
                items.append(el)
            if len(items) > 10000:
                break#
        self.items = items
        #self.items = [el for el in tqdm(self.items, desc='removing nonexistent files') if os.path.exists(el[1]['audio'])]

    def create_batches(self):
        np.random.seed(1234)
        indices = np.arange(len(self))
        indices = [np.random.permutation(indices[i:i+self.subgroup_shuffle_size]) for i in range(0, len(indices), self.subgroup_shuffle_size)]
        indices = np.concatenate(indices)
        indices = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        np.random.shuffle(indices)
        indices = np.concatenate(indices)
        self.items = [self.items[i] for i in indices]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio, txt = load_sample(self.pairs[self.items[idx][0]])
        id = self.items[idx][0]
        txt = txt['results'][-1]['alternatives'][0]['words']

        audio = rearrange(audio, '() f t -> t f')
        return audio, txt, id


def collate_fn(
        tokenizer:spm.SentencePieceProcessor,
        chunk_size:int = 2048,
        chunk_overlap:int = 192,
    ):
    pad_id = tokenizer.pad_id()
    bos_id = tokenizer.bos_id()

    def collate(batch): # move as much as possible from collate to dataset bcos datloader workers aren't taken advantage of here
        audio, txt, ids = zip(*batch)
       
        audio_lengths = torch.LongTensor([el.shape[0] for el in audio])
  
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=pad_id)
        audio = rearrange(audio, 'b t f -> b f t')
        
        txt_chunks = [chunk_text_json(text = el, chunk_size = chunk_size, chunk_overlap = chunk_overlap, spectogram_length = audio.shape[-1]) for el in txt]

        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        chunks = []
        culm_lengths_audio = torch.zeros_like(audio_lengths)

        for ix, el in enumerate(audio_chunks_):
            remove_mask = ~(culm_lengths_audio > audio_lengths)
            cur_chunks, cur_culm_lengths = el[remove_mask], culm_lengths_audio[remove_mask]
            cur_lengths = cur_chunks.shape[-1] - (cur_culm_lengths + cur_chunks.shape[-1] - audio_lengths[remove_mask] - chunk_overlap).clamp(0)
          
            enc_txt_chunks = [torch.LongTensor(tokenizer.encode(el[ix])) for i, el in enumerate(txt_chunks) if remove_mask[i]]
            enc_txt_chunks_lengths = torch.LongTensor([el.shape[0] for el in enc_txt_chunks])
            enc_txt_chunks = torch.nn.utils.rnn.pad_sequence(enc_txt_chunks, batch_first=True, padding_value=pad_id)

            chunks.append({
                'audio':cur_chunks,
                'txt':enc_txt_chunks,
                'txt_lengths':enc_txt_chunks_lengths,
                'audio_lengths':cur_lengths,
                'selection_mask':remove_mask,
                'cur_culm_lengths':cur_culm_lengths,
            })

            culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (chunk_overlap if ix != 0 else 0)
            
        return {
            'chunks': chunks,
            'total_audio_lengths': audio_lengths,
            'ids': ids,
        }

    return collate

class SimpleDataloader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        pairs:Dict[str, Dict[str, str]], 
        tokenizer:spm.SentencePieceProcessor, 
        skip_to:int = 0,
        batch_size:int = 5,
        chunk_size:int = 2048,
        chunk_overlap:int = 192,
    ):
        self.tokenizer = tokenizer
        self.dataset = SimpleDataset(
            pairs, 
            batch_size = batch_size,
            skip_to = skip_to, 
            subgroup_shuffle_size = 1000
        )
        super().__init__(
                self.dataset, 
                batch_size = batch_size, 
                shuffle = False, 
                num_workers = 0, 
                pin_memory = False, 
                collate_fn = collate_fn(
                    tokenizer = tokenizer,
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                )
            )


