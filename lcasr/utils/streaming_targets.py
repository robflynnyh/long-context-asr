import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from lcasr.utils.audio_tools import total_frames


IGNORE_ID = -100


def resolve_timed_words(text: Any) -> List[Dict[str, Any]]:
    if isinstance(text, dict):
        if "word_timestamps" in text:
            text = text["word_timestamps"]
        else:
            text = text["results"][-1]["alternatives"][0]["words"]

    if len(text) == 0:
        return []

    first = text[0]
    if {"startTime", "endTime", "word"}.issubset(first.keys()):
        return text
    if {"start", "end", "text"}.issubset(first.keys()):
        return text

    raise ValueError("unsupported transcript timestamp format")


def _word_fields(word: Dict[str, Any]) -> Tuple[float, float, str]:
    if "startTime" in word:
        return float(word["startTime"][:-1]), float(word["endTime"][:-1]), word["word"]
    return float(word["start"]), float(word["end"]), word["text"]


def filter_words_by_end_frame(text: Any, start_frame: int, end_frame: int) -> List[Dict[str, Any]]:
    filtered = []
    for word in resolve_timed_words(text):
        _, end_time, _ = _word_fields(word)
        word_end_frame = total_frames(end_time)
        if start_frame <= word_end_frame < end_frame:
            filtered.append(word)
    return filtered


def streaming_padding_frames(delay_seconds: float, buffer_seconds: float = 0.25) -> int:
    return max(0, total_frames(delay_seconds + buffer_seconds))


def pad_audio_for_streaming_delay(
    audio: torch.Tensor,
    lengths: torch.Tensor,
    delay_seconds: float,
    buffer_seconds: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pad_frames = streaming_padding_frames(delay_seconds, buffer_seconds)
    if pad_frames == 0:
        return audio, lengths
    audio = torch.nn.functional.pad(audio, (0, pad_frames), value=0.0)
    return audio, lengths + pad_frames


def build_streaming_frame_targets(
    transcripts: Sequence[Any],
    output_lengths: torch.Tensor,
    tokenizer: Any,
    subsampling_factor: int,
    delay_seconds: float = 2.0,
    chunk_start_frames: Optional[torch.Tensor] = None,
    silence_id: Optional[int] = None,
    ignore_id: int = IGNORE_ID,
) -> torch.Tensor:
    silence_id = tokenizer.vocab_size() if silence_id is None else silence_id
    device = output_lengths.device
    batch_size = len(transcripts)
    max_output_length = int(output_lengths.max().item())
    targets = torch.full(
        (batch_size, max_output_length),
        silence_id,
        dtype=torch.long,
        device=device,
    )

    if chunk_start_frames is None:
        chunk_start_frames = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        chunk_start_frames = chunk_start_frames.to(device=device, dtype=torch.long)

    delayed_frame_offset = total_frames(delay_seconds)

    for batch_idx, text in enumerate(transcripts):
        cursor = 0
        start_frame = int(chunk_start_frames[batch_idx].item())
        out_len = int(output_lengths[batch_idx].item())
        for word in resolve_timed_words(text):
            _, end_time, surface = _word_fields(word)
            delayed_frame = total_frames(end_time) + delayed_frame_offset - start_frame
            if delayed_frame < 0:
                continue

            out_pos = max(cursor, delayed_frame // subsampling_factor)
            if out_pos >= out_len:
                continue

            token_ids = tokenizer.encode(surface)
            for token_id in token_ids:
                if out_pos >= out_len:
                    break
                targets[batch_idx, out_pos] = int(token_id)
                out_pos += 1
            cursor = max(cursor, out_pos)

    padding_mask = torch.arange(max_output_length, device=device).expand(batch_size, -1)
    targets = targets.masked_fill(padding_mask >= output_lengths.unsqueeze(1), ignore_id)
    return targets
