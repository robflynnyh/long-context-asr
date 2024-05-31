import lcasr
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from omegaconf.omegaconf import OmegaConf
import traceback
from lcasr.utils.dataloading import VariableBatchSimpleDataloader, chunk_spectogram, chunk_text_json, reset_seen_ids
from lcasr.utils.hooks import add_debug_backwards_hooks
from lcasr.utils.scheduling import CosineLRScheduler, SequenceWarmupManager
from lcasr.utils.helpers import exists
from lcasr.utils.general import load_model, save_model, load_checkpoint, load_optimizer, get_model_class
from lcasr.utils.augmentation import SpecAugment
import resource
import time

from einops import rearrange
import numpy as np
import os
import wandb
from contextlib import nullcontext
from functools import partial

from torch.cuda.amp import GradScaler
from torch import autocast

from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
import random
random.seed(1234)


def save_utterances(
        args,
        dataloader, 
        tokenizer, 
        chunk_size=2048
    ):
    i, finished = -1, False
    dataloader_iter = iter(dataloader)
    total_recordings = dataloader.total_recordings()
    pad_id = tokenizer.pad_id()
    pbar = tqdm(total = len(dataloader), desc = f'Saving Utterances')
    while not finished:
        try:
            batch, i = next(dataloader_iter), i + 1
            pbar.update(1) if i > 0 else None
        except StopIteration:
            finished = True
            continue
        ################################

        audio, audio_lengths, txt, ids = batch

        audio_chunks_ = chunk_spectogram(spec = audio, chunk_size = chunk_size, chunk_overlap = 0)
        txt_chunks = [chunk_text_json(text = el, chunk_size = chunk_size, chunk_overlap = 0, spectogram_length = audio.shape[-1]) for el in txt] # becomes v slow for v large batch sizes !!

        del audio
        backwards_every_loss, steps_since_backwards = 0.0, 0
        chunks, culm_lengths_audio, nans_in_a_row = [], torch.zeros_like(audio_lengths), 0

        ################################
        for ix, el in enumerate(audio_chunks_):
            remove_mask = ~(culm_lengths_audio > audio_lengths)
            cur_chunks, cur_culm_lengths = el[remove_mask], culm_lengths_audio[remove_mask]
            cur_lengths = cur_chunks.shape[-1] - (cur_culm_lengths + cur_chunks.shape[-1] - audio_lengths[remove_mask] - 0).clamp(0)
          
            enc_txt_chunks = [torch.LongTensor(tokenizer.encode(el[ix])) for i, el in enumerate(txt_chunks) if remove_mask[i]]
            enc_txt_chunks_lengths = torch.LongTensor([el.shape[0] for el in enc_txt_chunks])
            enc_txt_chunks = torch.nn.utils.rnn.pad_sequence(enc_txt_chunks, batch_first=True, padding_value=pad_id)
            if enc_txt_chunks_lengths.max() == 0:
                continue # skip if none contain text (bad batch)

            torch.save({
                'id':f'{ids[0]}_{ix}',
                'audio':cur_chunks,
                'txt':enc_txt_chunks,
                'txt_lengths':enc_txt_chunks_lengths,
                'audio_lengths':cur_lengths,
            }, os.path.join(args.save_dir, f'{ids[0]}_{ix}.pt'))

            chunks.append({
                'id':f'{ids[0]}_{ix}', # f'{ids[0]}_{ix}
                'audio':cur_chunks,
                'txt':enc_txt_chunks,
                'txt_lengths':enc_txt_chunks_lengths,
                'audio_lengths':cur_lengths,
                'selection_mask':remove_mask,
                'cur_culm_lengths':cur_culm_lengths,
            })
            culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (0 if ix != 0 else 0)


def main(args):
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
   
    paired_data = lcasr.utils.audio_tools.load_json("/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json")


    # skip data up to step
    dataloader = VariableBatchSimpleDataloader(
        pairs = paired_data, 
        tokenizer = tokenizer, 
        batch_size = 1,
        chunk_size = 2048,
        chunk_overlap = 0,
    )

    save_utterances(
        args = args,
        dataloader = dataloader,
        tokenizer = tokenizer,
        chunk_size = 2048
    )





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-save', '--save_dir', type=str, help='Directory to save utterances to', default="/mnt/parscratch/users/acp21rjf/spotify_utterances")
    args = parser.parse_args()
    main(args)
      