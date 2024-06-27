import torch, numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List
from lcasr.utils.helpers import load_json, exists, load_pairs
from lcasr.utils.audio_tools import total_seconds
from lcasr.utils.augmentation import SpecAugment
from einops import rearrange
import sentencepiece as spm
import os
import time
import pandas as pd
import re

def chunk_spectogram( # TODO: speed up
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


def chunk_text_json( # TODO: speed up
        text: List[Dict[str, str]],
        chunk_size: int,
        chunk_overlap: int,
        spectogram_length: int,
        get_seconds: bool = False
    ):
    assert chunk_size > chunk_overlap, "chunk_size must be greater than chunk_overlap"
    
    text_remaining = text
    splits = []
    start_end_times = []
    for i in range(0, spectogram_length, chunk_size - chunk_overlap):
        c_start_pos, c_end_pos = i, i + chunk_size
        c_start_pos_sec, c_end_pos_sec = total_seconds(c_start_pos), total_seconds(c_end_pos)
        chunk_overlap_sec = total_seconds(chunk_overlap)
        c_text = []
        max_text_index = 0
        for i, el in enumerate(text_remaining):
            if float(el['startTime'][:-1]) >= c_start_pos_sec and float(el['endTime'][:-1]) <= c_end_pos_sec:
                c_text.append(el['word'])
            if float(el['endTime'][:-1]) < c_end_pos_sec - chunk_overlap_sec:
                max_text_index = i
            if float(el['endTime'][:-1]) > c_end_pos_sec:
                break
        text_remaining = text_remaining[max_text_index:]
        splits.append(" ".join(c_text))
        start_end_times.append((c_start_pos_sec, c_end_pos_sec))
    
    return splits if not get_seconds else (splits, start_end_times)


def chunk_text_and_speakers_json( # TODO: speed up
        text: List[Dict[str, str]],
        chunk_size: int,
        chunk_overlap: int,
        spectogram_length: int,
        get_seconds: bool = False
    ):
    assert chunk_size > chunk_overlap, "chunk_size must be greater than chunk_overlap"
    
    text_remaining = text
    splits = []
    speakers = []
    start_end_times = []
    for i in range(0, spectogram_length, chunk_size - chunk_overlap):
        c_start_pos, c_end_pos = i, i + chunk_size
        c_start_pos_sec, c_end_pos_sec = total_seconds(c_start_pos), total_seconds(c_end_pos)
        chunk_overlap_sec = total_seconds(chunk_overlap)
        c_text = []
        c_speakers = []
        max_text_index = 0
        for i, el in enumerate(text_remaining):
            if float(el['startTime'][:-1]) >= c_start_pos_sec and float(el['endTime'][:-1]) <= c_end_pos_sec:
                c_text.append(el['word'])
                c_speakers.append(el['speakerTag'])
            if float(el['endTime'][:-1]) < c_end_pos_sec - chunk_overlap_sec:
                max_text_index = i
            if float(el['endTime'][:-1]) > c_end_pos_sec:
                break
        text_remaining = text_remaining[max_text_index:]
        c_speakers = len(set(c_speakers))
        speakers.append(c_speakers)
        splits.append(" ".join(c_text))
        start_end_times.append((c_start_pos_sec, c_end_pos_sec))
    
    return (splits, speakers) if not get_seconds else (splits, speakers, start_end_times)

def chunk_text_json_with_speaker_change( # TODO: speed up
        text: List[Dict[str, str]],
        chunk_size: int,
        chunk_overlap: int,
        spectogram_length: int,
        get_seconds: bool = False,
        speaker_change_token: str = "¬"
    ):
    assert chunk_size > chunk_overlap, "chunk_size must be greater than chunk_overlap"
    
    text_remaining = text
    splits = []
    start_end_times = []
    for i in range(0, spectogram_length, chunk_size - chunk_overlap):
        c_start_pos, c_end_pos = i, i + chunk_size
        c_start_pos_sec, c_end_pos_sec = total_seconds(c_start_pos), total_seconds(c_end_pos)
        chunk_overlap_sec = total_seconds(chunk_overlap)
        c_text = []
        max_text_index = 0
        prev_speaker = None
        for i, el in enumerate(text_remaining):
            prev_speaker = el['speakerTag'] if prev_speaker is None else prev_speaker
            if float(el['startTime'][:-1]) >= c_start_pos_sec and float(el['endTime'][:-1]) <= c_end_pos_sec:
                if el['speakerTag'] != prev_speaker:
                    c_text.append(speaker_change_token)
                c_text.append(el['word'])
                prev_speaker = el['speakerTag']
            if float(el['endTime'][:-1]) < c_end_pos_sec - chunk_overlap_sec:
                max_text_index = i
            if float(el['endTime'][:-1]) > c_end_pos_sec:
                break
        text_remaining = text_remaining[max_text_index:]
        splits.append(" ".join(c_text))
        start_end_times.append((c_start_pos_sec, c_end_pos_sec))
    
    return splits if not get_seconds else (splits, start_end_times)


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


def reset_seen_ids(seen_ids:List[str], epoch:int):
    # take seen_ids that don't feature a epoch id and adds it to the string # this stops the dataloader from removeing seen_ids from previous epochs
    seen_ids = [f'epoch_{epoch}_{el}' if 'epoch_' not in el else el for el in seen_ids]
    return seen_ids

### for presegmented data
class Utterance_Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            utterance_folder:str,
            seen_ids:List[str] = [], # remove ids from dataset (i.e already trained on)
    ):
        super().__init__()
        self.utterance_folder = utterance_folder
        files = [el for el in os.listdir(utterance_folder) if el.endswith('.pt')]
        files_set = set(files)
        seen_ids_set = set([el + '.pt' for el in seen_ids])
        files = list(files_set - seen_ids_set) # remove seen_ids
        self.files = sorted([os.path.join(utterance_folder, el) for el in files])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data['id'], data['audio'], data['txt'], data['txt_lengths'], data['audio_lengths']

def utterance_collate_fn(batch):
    ids, audio, txt, txt_lengths, audio_lengths = zip(*batch)
    txt_lengths = torch.cat(txt_lengths)
    audio_lengths = torch.cat(audio_lengths)
    max_audio_length = audio_lengths.max()
    max_txt_length = txt_lengths.max()
    
    pad_amount = max_audio_length - audio_lengths
    audio = torch.cat([torch.nn.functional.pad(el, (0, pad_amount[i]), value=0) for i, el in enumerate(audio)], dim=0)
    pad_amount = max_txt_length - txt_lengths
    txt = torch.cat([torch.nn.functional.pad(el, (0, pad_amount[i]), value=0) for i, el in enumerate(txt)], dim=0)

    return {
        'ids':ids,
        'audio':audio,
        'text':txt,
        'text_lengths':txt_lengths,
        'audio_lengths':audio_lengths
    }
    

class Utterance_Dataloader(torch.utils.data.DataLoader):
    def __init__(
            self,
            utterance_folder:str,
            batch_size:int = 176,
            num_workers:int = 4,
            prefetch:int = 4,
            pin_memory:bool = True,
            shuffle:bool = True,
            seen_ids:List[str] = [],
            random_seed:int = 1234,
        ):
        torch.manual_seed(random_seed)
        self.dataset = Utterance_Dataset(utterance_folder, seen_ids = seen_ids)
        super().__init__(
            dataset = self.dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            pin_memory = pin_memory,
            prefetch_factor = prefetch,
            collate_fn = utterance_collate_fn,
        )


    def total_recordings(self):
        return len(self.dataset.files)

    def __len__(self):
        return len(self.dataset) 

### for presegmented data


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            pairs:Dict[str, Dict[str, str]],
            batch_size:int = 8,
            subgroup_shuffle_size:int = 2000,
            skip_to:int = 0, # deprecated
            random_seed:int = 1234,
            seen_ids:List[str] = [], # remove ids from dataset (i.e already trained on)
        ):
        self.batch_size = batch_size
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.random_seed = random_seed
  
        self.pairs = pd.DataFrame(list(pairs.values()))
        self.pairs['id'] = list(pairs.keys())
        # remove ids
        self.pairs = self.pairs[~self.pairs['id'].isin(seen_ids)]

        # sort pairs by duration
        self.pairs = self.pairs.sort_values(by='duration')
        self.pairs = self.pairs.reset_index(drop=True) # reset index drop means old index is not added as a column

        self.create_batches()
        # trim to skip_to
        self.pairs = self.pairs.iloc[skip_to:].reset_index(drop=True) # deprecated in favour of seen_ids !!
       

    def create_batches(self):
        np.random.seed(self.random_seed)
        indices = np.arange(len(self))
        indices = [np.random.permutation(indices[i:i+self.subgroup_shuffle_size]) for i in range(0, len(indices), self.subgroup_shuffle_size)]
        indices = np.concatenate(indices)
        indices = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
        np.random.shuffle(indices)
        indices = np.concatenate(indices)
        self.pairs = self.pairs.iloc[indices].reset_index(drop=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio, txt = load_sample({'audio': self.pairs['audio'][idx], 'txt': self.pairs['txt'][idx]})
        id = self.pairs['id'][idx]
        txt = txt['results'][-1]['alternatives'][0]['words']
        audio = rearrange(audio, '() f t -> t f')
        return audio, txt, id


def collate_fn():
    def collate(batch):
        audio, txt, ids = zip(*batch)
        audio_lengths = torch.LongTensor([el.shape[0] for el in audio])
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
        audio = rearrange(audio, 'b t f -> b f t')
        return audio, audio_lengths, txt, ids
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
        num_workers:int = 0,
        pin_memory:bool = False,
        prefetch:int = None,
        random_seed=1234,
        subgroup_shuffle_size:int = 2000,
        seen_ids:List[str] = [],
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.skip_to = skip_to

        dataset = SimpleDataset(
                    pairs, 
                    batch_size = batch_size,
                    skip_to = skip_to, 
                    subgroup_shuffle_size = subgroup_shuffle_size,
                    random_seed = random_seed,
                    seen_ids = seen_ids,
        )
        super().__init__(
                dataset = dataset,
                batch_size = batch_size, 
                shuffle = False, 
                num_workers = num_workers, 
                pin_memory = pin_memory, 
                collate_fn = kwargs.get('collate_fn', collate_fn()),
                prefetch_factor = prefetch if num_workers > 0 else None,
            )
            

class VariableBatchSimpleDataloader():
    def __init__(
        self, 
        pairs:Dict[str, Dict[str, str]], 
        tokenizer:spm.SentencePieceProcessor, 
        skip_to:int = 0,
        batch_size:int = 5,
        chunk_size:int = 2048,
        chunk_overlap:int = 192,
        num_workers:int = 0,
        pin_memory:bool = False,
        prefetch:int = None,
        random_seed=1234,
        subgroup_shuffle_size:int = 2000,
        seen_ids:List[str] = [],
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        self.subgroup_shuffle_size = subgroup_shuffle_size
        self.pairs = pairs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch = prefetch
        self.random_seed = random_seed

        self.dataloader = SimpleDataloader(
            pairs = pairs,
            tokenizer = tokenizer,
            skip_to = skip_to,
            batch_size = batch_size,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            num_workers = num_workers,
            pin_memory = pin_memory,
            prefetch = prefetch,
            subgroup_shuffle_size = subgroup_shuffle_size,
            random_seed = random_seed,
            seen_ids = seen_ids,
            **kwargs
        )

    def update(
            self, 
            batch_size:int, 
            seen_ids:List[str]=[],
            random_seed:int='same'
        ):
        self.batch_size = batch_size
        self.dataloader = SimpleDataloader(
            pairs = self.pairs,
            tokenizer = self.tokenizer,
            batch_size = batch_size,
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            prefetch = self.prefetch,
            random_seed = self.random_seed if random_seed == 'same' else random_seed,
            seen_ids = seen_ids,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def total_recordings(self):
        return len(self.pairs.keys())

    def __len__(self):
        return len(self.dataloader) 







