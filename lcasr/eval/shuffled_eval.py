# inspired by: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Streaming_ASR.ipynb

import numpy as np
import torch
from lcasr.utils.audio_tools import total_frames, total_seconds
from typing import List, Dict, Tuple
from tqdm import tqdm
from lcasr.models.sconformer_xl import SCConformerXL

def prepare_chunks(spec, seq_len, overlap = 0):
    spec_n = spec.shape[-1]
    last_ulen, kill_next = None, False

    if spec_n <= seq_len:
        return {0: spec}, [0]

    training_data = {}
    for i in range(0, spec_n, seq_len-overlap):
        audio_chunk = spec[:, :, i:i+seq_len] # [B, C, T]
        u_len = audio_chunk.shape[-1]
        if kill_next:
            break
        elif last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len
        training_data[i] = audio_chunk
    return training_data, list(training_data.keys())

@torch.no_grad()
def fetch_logits(args, model:SCConformerXL, spec:torch.Tensor, distracter_spec:torch.Tensor, distracter_spec_chunks_len:int, seq_len:int, window_len:int, buffer_len:int, tokenizer, use_tqdm=True):
    '''
    args: argparse.Namespace
    model: instance of CTC based model
    spec: spectrogram tensor
    seq_len: sequence length
    overlap: here we use overlap to refer to the chunk size that is used for transcription, the buffer is equal to seq_len - overlap
    tokenizer: tokenizer instance
    use_tqdm: bool, whether to use tqdm or not
    '''

    spec_n = spec.shape[-1]
    downsampling_factor = model.subsampling.subsampling_factor
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']
 
    if seq_len > spec_n:
        seq_len = spec_n
        overlap = 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']
    
    #assert overlap == 0 or cache_len == 0, 'Cannot use overlap and cache_len at the same time'

    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'

    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    logit_position = 0

    chunk_size = window_len
    
    chunk_i_start = 0
    chunk_i_end = chunk_i_start + chunk_size

    finished = False
    positions = []
    while not finished:
        spec_start = chunk_i_start - buffer_len // 2
        spec_end = chunk_i_end + buffer_len // 2

        if spec_start < 0:
            spec_start = 0
            spec_end = chunk_size + buffer_len
        elif spec_end > spec_n:
            spec_end = spec_n
            spec_start = spec_end - chunk_size - buffer_len

        positions.append({
            'buffer_start_frame': spec_start,
            'buffer_end_frame': spec_end,
            'chunk_start_frame': chunk_i_start,
            'chunk_end_frame': chunk_i_end
        })
        chunk_i_start += chunk_size
        chunk_i_end += chunk_size
        if chunk_i_end >= spec_n:
            chunk_i_end = spec_n
        if chunk_i_start >= spec_n:
            finished = True

    distracter_spec_chunks = list(prepare_chunks(spec = distracter_spec, seq_len = distracter_spec_chunks_len, overlap = 0)[0].values())

    
    logit_position = 0
    for i, pos in tqdm(enumerate(positions), total=len(positions)) if use_tqdm else enumerate(positions):
        buffer_start, buffer_end = pos['buffer_start_frame'], pos['buffer_end_frame']
        chunk_start, chunk_end = pos['chunk_start_frame'], pos['chunk_end_frame']
        #print(buffer_end-buffer_start, chunk_end-chunk_start)
        audio_chunk = spec[:, :, buffer_start:buffer_end].clone()
        
        audio_chunk_len = audio_chunk.shape[-1]
        to_add = spec_n - audio_chunk_len
        chunks_to_add = []
        while to_add > 0:
            distracter_chunk = distracter_spec_chunks[np.random.randint(0, len(distracter_spec_chunks))].clone()
            distracter_chunk_len = distracter_chunk.shape[-1]
            chunks_to_add.append(distracter_chunk)
            to_add -= distracter_chunk_len
        
        add_before = torch.cat(chunks_to_add[:len(chunks_to_add)//2], dim=-1)
        add_after = torch.cat(chunks_to_add[len(chunks_to_add)//2:], dim=-1)
        add_before_len, add_after_len = add_before.shape[-1], add_after.shape[-1]
        audio_chunk = torch.cat([add_before, audio_chunk, add_after], dim=-1)

        # normalise chunk
        # audio_chunk = (audio_chunk - audio_chunk.mean(-1, keepdim=True)) / (audio_chunk.std(-1, keepdim=True) + 1e-5)
        audio_chunk = audio_chunk.to(model.device)
        out = model(audio_chunk)
        rel_chunk_start, rel_chunk_end = chunk_start - buffer_start, chunk_end - buffer_start # relative to start position of buffer
        
        spec_in_size = audio_chunk.shape[-1]
        logits = out['final_posteriors'].detach().cpu()
        logit_size = logits.shape[-2]
        downsampled_by = spec_in_size / logit_size
  

        downsampled_before, downsampled_window = int(add_before_len / downsampled_by), int(audio_chunk_len / downsampled_by)
        audio_chunk = logits[:, downsampled_before:downsampled_before+downsampled_window, :].clone()
        
        logits = audio_chunk
        logit_size = logits.shape[-2]

        rel_chunk_start_ds, rel_chunk_end_ds = int(rel_chunk_start / downsampled_by), int(rel_chunk_end / downsampled_by)
        all_logits[:, logit_position + rel_chunk_start_ds:logit_position + rel_chunk_end_ds, :] += logits[:, rel_chunk_start_ds:rel_chunk_end_ds, :]
        logit_count[:, logit_position + rel_chunk_start_ds:logit_position + rel_chunk_end_ds, :] += 1
        logit_position += rel_chunk_end_ds - rel_chunk_start_ds

    assert logit_count.max() == 1, 'logit_count should not be greater than 1'
    B,N,C = all_logits.shape
    logits = all_logits[logit_count.sum(dim=-1) != 0]
    logits = logits.reshape(B,-1,C)
   
    return logits.squeeze(0).numpy()
