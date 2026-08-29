import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from lcasr.utils.audio_tools import total_frames


IGNORE_ID = -100
TargetEvent = Tuple[int, int]


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


def filter_words_by_frame_overlap(text: Any, start_frame: int, end_frame: int) -> List[Dict[str, Any]]:
    filtered = []
    for word in resolve_timed_words(text):
        start_time, end_time, _ = _word_fields(word)
        word_start_frame = total_frames(start_time)
        word_end_frame = total_frames(end_time)
        if word_start_frame < end_frame and word_end_frame > start_frame:
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
    subsampling_factor: Optional[int],
    delay_seconds: float = 2.0,
    chunk_start_frames: Optional[torch.Tensor] = None,
    output_length_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    silence_id: Optional[int] = None,
    ignore_id: int = IGNORE_ID,
) -> torch.Tensor:
    if output_length_fn is None and subsampling_factor is None:
        raise ValueError("build_streaming_frame_targets requires subsampling_factor or output_length_fn")
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
    output_position_cache: Dict[int, int] = {}

    def frame_to_output_position(frame: int) -> int:
        frame = max(0, int(frame))
        if output_length_fn is None:
            return frame // int(subsampling_factor)
        if frame == 0:
            return 0
        if frame not in output_position_cache:
            frame_tensor = torch.tensor([frame], dtype=torch.long, device=device)
            output_position_cache[frame] = int(output_length_fn(frame_tensor).reshape(-1)[0].item())
        return output_position_cache[frame]

    for batch_idx, text in enumerate(transcripts):
        cursor = 0
        start_frame = int(chunk_start_frames[batch_idx].item())
        chunk_start_output = frame_to_output_position(start_frame)
        out_len = int(output_lengths[batch_idx].item())
        for word in resolve_timed_words(text):
            _, end_time, surface = _word_fields(word)
            delayed_frame = total_frames(end_time) + delayed_frame_offset
            delayed_output = frame_to_output_position(delayed_frame)

            out_pos = max(cursor, delayed_output)

            token_ids = tokenizer.encode(surface)
            for token_offset, token_id in enumerate(token_ids):
                local_pos = out_pos + token_offset - chunk_start_output
                if local_pos < 0:
                    continue
                if local_pos >= out_len:
                    break
                targets[batch_idx, local_pos] = int(token_id)
            cursor = max(cursor, out_pos + len(token_ids))

    padding_mask = torch.arange(max_output_length, device=device).expand(batch_size, -1)
    targets = targets.masked_fill(padding_mask >= output_lengths.unsqueeze(1), ignore_id)
    return targets


def build_streaming_target_events(
    transcript: Any,
    tokenizer: Any,
    subsampling_factor: Optional[int],
    delay_seconds: float = 2.0,
    output_length_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[TargetEvent]:
    if output_length_fn is None and subsampling_factor is None:
        raise ValueError("build_streaming_target_events requires subsampling_factor or output_length_fn")

    delayed_frame_offset = total_frames(delay_seconds)
    output_position_cache: Dict[int, int] = {}

    def frame_to_output_position(frame: int) -> int:
        frame = max(0, int(frame))
        if output_length_fn is None:
            return frame // int(subsampling_factor)
        if frame == 0:
            return 0
        if frame not in output_position_cache:
            frame_tensor = torch.tensor([frame], dtype=torch.long)
            output_position_cache[frame] = int(output_length_fn(frame_tensor).reshape(-1)[0].item())
        return output_position_cache[frame]

    events: List[TargetEvent] = []
    cursor = 0
    for word in resolve_timed_words(transcript):
        _, end_time, surface = _word_fields(word)
        delayed_frame = total_frames(end_time) + delayed_frame_offset
        delayed_output = frame_to_output_position(delayed_frame)
        out_pos = max(cursor, delayed_output)
        token_ids = tokenizer.encode(surface)
        for token_offset, token_id in enumerate(token_ids):
            events.append((out_pos + token_offset, int(token_id)))
        cursor = max(cursor, out_pos + len(token_ids))
    return events


def build_streaming_frame_targets_from_events(
    event_sequences: Sequence[Sequence[TargetEvent]],
    output_lengths: torch.Tensor,
    chunk_start_frames: torch.Tensor,
    tokenizer: Any,
    subsampling_factor: Optional[int],
    delay_seconds: float = 2.0,
    output_length_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    silence_id: Optional[int] = None,
    ignore_id: int = IGNORE_ID,
    event_offsets: Optional[Sequence[int]] = None,
    return_event_offsets: bool = False,
):
    if output_length_fn is None and subsampling_factor is None:
        raise ValueError("build_streaming_frame_targets_from_events requires subsampling_factor or output_length_fn")
    silence_id = tokenizer.vocab_size() if silence_id is None else silence_id
    device = output_lengths.device
    batch_size = len(event_sequences)
    max_output_length = int(output_lengths.max().item()) if output_lengths.numel() else 0
    targets = torch.full(
        (batch_size, max_output_length),
        silence_id,
        dtype=torch.long,
        device=device,
    )
    chunk_start_frames = chunk_start_frames.to(device=device, dtype=torch.long)
    output_position_cache: Dict[int, int] = {}

    def frame_to_output_position(frame: int) -> int:
        frame = max(0, int(frame))
        if output_length_fn is None:
            return frame // int(subsampling_factor)
        if frame == 0:
            return 0
        if frame not in output_position_cache:
            frame_tensor = torch.tensor([frame], dtype=torch.long, device=device)
            output_position_cache[frame] = int(output_length_fn(frame_tensor).reshape(-1)[0].item())
        return output_position_cache[frame]

    offsets = [0] * batch_size if event_offsets is None else [int(offset) for offset in event_offsets]
    next_offsets: List[int] = []
    for batch_idx, events in enumerate(event_sequences):
        chunk_start_output = frame_to_output_position(int(chunk_start_frames[batch_idx].item()))
        out_len = int(output_lengths[batch_idx].item())
        chunk_end_output = chunk_start_output + out_len
        event_idx = max(0, offsets[batch_idx])
        while event_idx < len(events) and int(events[event_idx][0]) < chunk_start_output:
            event_idx += 1
        write_idx = event_idx
        while write_idx < len(events):
            output_pos, token_id = events[write_idx]
            output_pos = int(output_pos)
            if output_pos >= chunk_end_output:
                break
            local_pos = output_pos - chunk_start_output
            if 0 <= local_pos < out_len:
                targets[batch_idx, local_pos] = int(token_id)
            write_idx += 1
        next_offsets.append(write_idx)

    if max_output_length > 0:
        padding_mask = torch.arange(max_output_length, device=device).expand(batch_size, -1)
        targets = targets.masked_fill(padding_mask >= output_lengths.unsqueeze(1), ignore_id)
    if return_event_offsets:
        return targets, next_offsets
    return targets
