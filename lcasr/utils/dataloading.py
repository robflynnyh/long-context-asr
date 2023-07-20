import torch, numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, List
from lcasr.utils.helpers import load_json, exists, load_pairs
from lcasr.utils.audio_tools import total_seconds
from einops import rearrange
import sentencepiece as spm
import os
import time

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

def chunk_text_json____( # legacy
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

def chunk_text_json( # legacy
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


# def __chunk_eq_test__(text, chunk_size, chunk_overlap, spectogram_length, timeit=False):
#     '''equivalence test between legacy and new chunk_text_json functions'''
#     a_start = time.time() if timeit else None
#     a = ___chunk_text_json___(text, chunk_size, chunk_overlap, spectogram_length, get_seconds=False)
#     if timeit:
#         a_end, b_start = time.time(), time.time()
#     b = chunk_text_json(text, chunk_size, chunk_overlap, spectogram_length, get_seconds=False)
#     b_end = time.time() if timeit else None
#     all_a = "#".join(a)
#     all_b = "#".join(b)
#     assert all_a == all_b, f"FAIL\nall_a: {all_a}\nall_b: {all_b}"
#     print("PASS")
#     if timeit:
#         print(f"legacy: {a_end - a_start}\nnew: {b_end - b_start}")

# def run_chunk_eq_test():
#     pairs = load_pairs()
#     text_f = [pairs[k]['txt'] for k in list(pairs.keys())[:10000]]
#     for i in tqdm(range(len(text_f)), total=len(text_f)):
#         json_tx = load_json(text_f[i])
#         txt = json_tx['results'][-1]['alternatives'][0]['words']
#         __chunk_eq_test__(txt, 2048, 2000, 100000, timeit=True)

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

    def collate(batch): # move as much as possible from collate to dataset bcos datloader workers aren't taken advantage of here (I don't think??)
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer
        
        super().__init__(
                dataset = SimpleDataset(
                    pairs, 
                    batch_size = batch_size,
                    skip_to = skip_to, 
                    subgroup_shuffle_size = 1000
                ), 
                batch_size = batch_size, 
                shuffle = False, 
                num_workers = 0, 
                pin_memory = False, 
                collate_fn = lambda x: x,
            )


