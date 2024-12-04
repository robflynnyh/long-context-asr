# uses code from: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch import functional as F
from lcasr.utils.audio_tools import total_frames, total_seconds
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
from lcasr.models.sconformer_xl import SCConformerXL
from .buffered_transcription import fetch_logits
import sentencepiece as spm
from dataclasses import dataclass


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

# Merge the labels
@dataclass
class Segment:
    labels: Union[List[int], str]
    tokenized_transcript_indexes:List[int]
    start: int
    end: int
    scores: List[float]

    def __repr__(self):
        return f"{self.labels}\t({self.scores}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, tokens):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                [tokens[path[i1].token_index]],
                [path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                [score],
            )
        )
        i1 = i2
    return segments

def create_segments_of_length(segments:List[Segment], block_size_seconds, seconds_per_frame, tokenizer:spm.SentencePieceProcessor):
    new_segments = []
    delta = 0.0
    stack = []
    i = 0
    word = []

    def popstack(stack, new_segments):
        if len(stack) > 0:
            label_list,token_index_list,scores = [],[],[]
            for prev in stack: 
                label_list.extend(prev.labels)
                token_index_list.extend(prev.tokenized_transcript_indexes)
                scores.extend(prev.scores)
            new_segments.append(Segment(
                labels=label_list,
                tokenized_transcript_indexes=token_index_list,
                start=stack[0].start,
                end=stack[-1].end,
                scores=scores
            ))
            stack = []
        return stack, new_segments

    def popword(word, stack, new_segments, delta):
        if len(word) > 0:
            end_time = word[-1].end*seconds_per_frame
            if (end_time - delta) > block_size_seconds:
                stack, new_segments = popstack(stack, new_segments) 
                delta += block_size_seconds
            stack.extend(word)
            word = []
        return word, stack, new_segments, delta

    while i < len(segments):
        cur_segment = segments[i]
        assert len(cur_segment.labels) == 1, 'original segment must be single token level'
        segment_piece = tokenizer.IdToPiece(cur_segment.labels[0])
        if segment_piece.startswith('▁'):
            word, stack, new_segments, delta = popword(word, stack, new_segments, delta)
        word.append(cur_segment)
        i+=1
    _, stack, new_segments, _ = popword(word, stack, new_segments, delta)
    _, new_segments = popstack(stack, new_segments)

    return new_segments

def force_align(
        logits:Tensor,
        transcript:str,
        tokenizer:spm.SentencePieceProcessor,
        seconds_per_frame:float,
        block_sizes_seconds:float=10.24,
    ):
    print(seconds_per_frame, 1)
    tokens = tokenizer.Encode(transcript)
    print(transcript)
    print(logits.shape)
    logits = torch.as_tensor(logits)
    trellis = get_trellis(emission=logits, tokens=tokens)
    path = backtrack(trellis, logits, tokens)

    segments = merge_repeats(path, tokens=tokens)
    segments = create_segments_of_length(segments, block_size_seconds=block_sizes_seconds, seconds_per_frame=seconds_per_frame, tokenizer=tokenizer)
    for seg in segments:
        print(seg.start, seg.end)
        print(tokenizer.IdToPiece(seg.labels))
        print('---')

# python tts_context_eval.py -c /mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb/n_seq_sched_16384_rp_1/step_105360.pt  -seq 16384 -overlap 0