# inspired by: https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Streaming_ASR.ipynb

import numpy as np
import torch
from lcasr.utils.audio_tools import total_frames, total_seconds
from typing import List, Dict, Tuple
from tqdm import tqdm
from lcasr.models.sconformer_xl import SCConformerXL

# A simple iterator class to return successive chunks of samples
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
    logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))    
    logit_position = 0

    chunk_size = seq_len - overlap

    chunk_i_start = 0
    chunk_i_end = chunk_i_start + chunk_size



