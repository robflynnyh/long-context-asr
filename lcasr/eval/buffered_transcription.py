# inspired by: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Streaming_ASR.ipynb

import numpy as np
import torch
from lcasr.utils.audio_tools import total_frames, total_seconds
from typing import List, Dict, Tuple
from tqdm import tqdm
from lcasr.models.sconformer_xl import SCConformerXL

@torch.no_grad()
def fetch_logits(args, model:SCConformerXL, spec:torch.Tensor, seq_len:int, overlap:int, tokenizer, use_tqdm=True):
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
    logit_position = 0

    chunk_size = seq_len - overlap

    chunk_i_start = 0
    chunk_i_end = chunk_i_start + chunk_size

    finished = False
    positions = []
    while not finished:
        spec_start = chunk_i_start - overlap // 2
        spec_end = chunk_i_end + overlap // 2

        if spec_start < 0:
            spec_start = 0
            spec_end = seq_len
        elif spec_end > spec_n:
            spec_end = spec_n
            spec_start = spec_end - seq_len

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
    
    logit_position = 0
    for i, pos in tqdm(enumerate(positions), total=len(positions)) if use_tqdm else enumerate(positions):
        buffer_start, buffer_end = pos['buffer_start_frame'], pos['buffer_end_frame']
        chunk_start, chunk_end = pos['chunk_start_frame'], pos['chunk_end_frame']
        audio_chunk = spec[:, :, buffer_start:buffer_end]
        audio_chunk = audio_chunk.to(model.device)
        out = model(audio_signal = audio_chunk)
        rel_chunk_start, rel_chunk_end = chunk_start - buffer_start, chunk_end - buffer_start # relative to start position of buffer
        buffer_size = audio_chunk.shape[-1]
        logits = out['final_posteriors'].detach().cpu()
        logit_size = logits.shape[-2]
        downsampled_by = buffer_size / logit_size

        rel_chunk_start_ds, rel_chunk_end_ds = int(rel_chunk_start / downsampled_by), int(rel_chunk_end / downsampled_by)
        all_logits[:, logit_position + rel_chunk_start_ds:logit_position + rel_chunk_end_ds, :] += logits[:, rel_chunk_start_ds:rel_chunk_end_ds, :]
        logit_position += rel_chunk_end_ds - rel_chunk_start_ds

    all_logits = all_logits[:, :logit_position, :]
   
    return all_logits.squeeze(0).numpy()
