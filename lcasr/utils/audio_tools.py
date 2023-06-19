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
    '''converts number of frames to seconds'''
    return (spectogram_length * 160) / 16000

def total_frames(seconds:float) -> int:
    '''inverse of total_seconds'''
    return int((seconds * 16000) / 160) 

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

def chunk_text_json(
        text: List[Dict[str, str]],
        chunk_size: int,
        chunk_overlap: int,
        spectogram_length: int,
    ):
    assert chunk_size > chunk_overlap, "chunk_size must be greater than chunk_overlap"
    
    splits = []
    for i in range(0, spectogram_length, chunk_size - chunk_overlap):
        c_start_pos, c_end_pos = i, i + chunk_size
        c_start_pos_sec, c_end_pos_sec = total_seconds(c_start_pos), total_seconds(c_end_pos)
        c_text = [el['word'] for el in text if float(el['startTime'][:-1]) >= c_start_pos_sec and float(el['endTime'][:-1]) <= c_end_pos_sec]
        splits.append(" ".join(c_text))
        
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
    def __init__(self, pairs:Dict[str, Dict[str, str]]):
        self.pairs = pairs
        self.keys = list(pairs.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        audio, txt = load_sample(self.pairs[self.keys[idx]])
        txt = txt['results'][-1]['alternatives'][0]['words']

        audio = rearrange(audio, '() f t -> t f')
        return audio, txt


def collate_fn(
        tokenizer:spm.SentencePieceProcessor,
        chunk_size:int = 2048,
        chunk_overlap:int = 192,
    ):
    pad_id = tokenizer.pad_id()
    bos_id = tokenizer.bos_id()

    def collate(batch):
        audio, txt = zip(*batch)
       
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
            })

            culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (chunk_overlap if ix != 0 else 0)
            
        return {
            'chunks': chunks,
            'total_audio_lengths': audio_lengths,
        }

    return collate

class SimpleDataloader(torch.utils.data.DataLoader):
    def __init__(
        self, 
        pairs:Dict[str, Dict[str, str]], 
        tokenizer:spm.SentencePieceProcessor, 
        batch_size:int = 5,
        chunk_size:int = 2048,
        chunk_overlap:int = 192,
    ):
        self.dataset = SimpleDataset(pairs)
        super().__init__(
                self.dataset, 
                batch_size = batch_size, 
                shuffle = True, 
                num_workers = 0, 
                pin_memory = False, 
                collate_fn = collate_fn(
                    tokenizer = tokenizer,
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                )
            )


