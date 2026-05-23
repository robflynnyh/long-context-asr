import argparse
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lcasr
import torch
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from lcasr.components.positional_encodings import apply_rotary
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.dataloading import VariableBatchSimpleDataloader, reset_seen_ids
from lcasr.utils.general import get_model_class, load_checkpoint, load_model, load_optimizer
from lcasr.utils.streaming_targets import (
    filter_words_by_frame_overlap,
    pad_audio_for_streaming_delay,
    resolve_timed_words,
)

try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    EnglishTextNormalizer = None


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"invalid dtype: {dtype}")


def normalize_text(text: str, normalizer: Any = None) -> str:
    if normalizer is not None:
        text = normalizer(text)
    return " ".join(text.lower().strip().split())


def word_surface(word: Dict[str, Any]) -> str:
    if "word" in word:
        return str(word["word"])
    if "text" in word:
        return str(word["text"])
    return ""


def reference_words(transcript: Sequence[Dict[str, Any]], normalizer: Any = None) -> str:
    return normalize_text(" ".join(word_surface(word) for word in transcript), normalizer)


def timed_reference_words(transcript: Sequence[Dict[str, Any]], normalizer: Any = None) -> List[Dict[str, Any]]:
    timed_words = []
    for word in resolve_timed_words(transcript):
        surface = normalize_text(word_surface(word), normalizer)
        if not surface:
            continue
        end_time = float(str(word["endTime"])[:-1]) if "endTime" in word else float(word["end"])
        for token in surface.split():
            timed_words.append({"word": token, "time": end_time})
    return timed_words


def _decode_token_group(tokenizer: Any, token_ids: Sequence[int], normalizer: Any = None) -> List[str]:
    if len(token_ids) == 0:
        return []
    text = tokenizer.decode([int(token_id) for token_id in token_ids])
    return normalize_text(text, normalizer).split()


def prediction_words_with_times(
    model: torch.nn.Module,
    tokenizer: Any,
    prediction_ids: Sequence[int],
    chunk_start_frames: int,
    subsampling_factor: int,
    normalizer: Any = None,
) -> List[Dict[str, Any]]:
    words = []
    current_tokens = []
    current_end_time = None
    has_piece_api = hasattr(tokenizer, "id_to_piece")

    for frame_idx, token_id in enumerate(prediction_ids):
        token_id = int(token_id)
        if token_id == model.get_silence_id():
            continue

        token_time = total_seconds_from_frames(chunk_start_frames + frame_idx * subsampling_factor)
        starts_new_word = False
        if has_piece_api:
            piece = tokenizer.id_to_piece(token_id)
            starts_new_word = piece.startswith("▁") and len(current_tokens) > 0

        if starts_new_word:
            for word in _decode_token_group(tokenizer, current_tokens, normalizer=normalizer):
                words.append({"word": word, "time": current_end_time})
            current_tokens = []

        current_tokens.append(token_id)
        current_end_time = token_time

        if not has_piece_api:
            for word in _decode_token_group(tokenizer, [token_id], normalizer=normalizer):
                words.append({"word": word, "time": token_time})
            current_tokens = []
            current_end_time = None

    if current_tokens:
        for word in _decode_token_group(tokenizer, current_tokens, normalizer=normalizer):
            words.append({"word": word, "time": current_end_time})
    return words


def total_seconds_from_frames(frames: int) -> float:
    return float(frames) * 160.0 / 16000.0


def decode_model_prediction_ids(
    model: torch.nn.Module,
    tokenizer: Any,
    prediction_ids: Iterable[int],
    max_tokens: Optional[int] = None,
    normalizer: Any = None,
) -> str:
    text = model._decode_prediction_ids(
        tokenizer,
        prediction_ids,
        max_tokens=max_tokens,
    )
    return normalize_text(text, normalizer)


def word_alignment(ref_words: Sequence[str], hyp_words: Sequence[str]) -> List[Tuple[str, Optional[int], Optional[int]]]:
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]
    back: List[List[Optional[Tuple[str, Optional[int], Optional[int]]]]] = [[None] * cols for _ in range(rows)]

    for ref_idx in range(1, rows):
        dp[ref_idx][0] = ref_idx
        back[ref_idx][0] = ("delete", ref_idx - 1, None)
    for hyp_idx in range(1, cols):
        dp[0][hyp_idx] = hyp_idx
        back[0][hyp_idx] = ("insert", None, hyp_idx - 1)

    for ref_idx in range(1, rows):
        for hyp_idx in range(1, cols):
            same = ref_words[ref_idx - 1] == hyp_words[hyp_idx - 1]
            diag_cost = dp[ref_idx - 1][hyp_idx - 1] + (0 if same else 1)
            delete_cost = dp[ref_idx - 1][hyp_idx] + 1
            insert_cost = dp[ref_idx][hyp_idx - 1] + 1
            best = min(diag_cost, delete_cost, insert_cost)
            dp[ref_idx][hyp_idx] = best
            if diag_cost == best:
                back[ref_idx][hyp_idx] = ("equal" if same else "substitute", ref_idx - 1, hyp_idx - 1)
            elif delete_cost == best:
                back[ref_idx][hyp_idx] = ("delete", ref_idx - 1, None)
            else:
                back[ref_idx][hyp_idx] = ("insert", None, hyp_idx - 1)

    alignment = []
    ref_idx = len(ref_words)
    hyp_idx = len(hyp_words)
    while ref_idx > 0 or hyp_idx > 0:
        op = back[ref_idx][hyp_idx]
        if op is None:
            break
        alignment.append(op)
        if op[0] in {"equal", "substitute"}:
            ref_idx -= 1
            hyp_idx -= 1
        elif op[0] == "delete":
            ref_idx -= 1
        else:
            hyp_idx -= 1
    alignment.reverse()
    return alignment


def weighted_error_rewards(
    hypotheses: List[str],
    references: List[str],
    hypothesis_word_times: Optional[List[List[Dict[str, Any]]]] = None,
    reference_word_times: Optional[List[List[Dict[str, Any]]]] = None,
    late_word_tolerance_seconds: Optional[float] = None,
    late_word_penalty_per_second: float = 0.25,
    late_word_penalty_max: float = 1.0,
    wer_weight: float = 0.7,
    cer_weight: float = 0.3,
    reward_offset: float = 1.0,
    reward_scale: float = 1.0,
    reward_min: Optional[float] = 0.0,
    reward_max: Optional[float] = None,
    reward_positive_threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    weight_sum = wer_weight + cer_weight
    if weight_sum <= 0:
        raise ValueError("reward weights must sum to a positive value")
    if late_word_penalty_per_second < 0:
        raise ValueError("late_word_penalty_per_second must be non-negative")
    if late_word_penalty_max < 0:
        raise ValueError("late_word_penalty_max must be non-negative")
    wer_weight = wer_weight / weight_sum
    cer_weight = cer_weight / weight_sum

    rewards = []
    late_correct_words = 0
    late_correct_penalty = 0.0
    matched_words = 0
    for idx, (hyp, ref) in enumerate(zip(hypotheses, references)):
        wer, *_ = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)
        if (
            late_word_tolerance_seconds is not None
            and hypothesis_word_times is not None
            and reference_word_times is not None
            and idx < len(hypothesis_word_times)
            and idx < len(reference_word_times)
        ):
            ref_timed = reference_word_times[idx]
            hyp_timed = hypothesis_word_times[idx]
            alignment = word_alignment(
                [str(word["word"]) for word in ref_timed],
                [str(word["word"]) for word in hyp_timed],
            )
            ref_count = max(len(ref_timed), 1)
            late_count = 0
            late_penalty = 0.0
            for op, ref_idx, hyp_idx in alignment:
                if op != "equal" or ref_idx is None or hyp_idx is None:
                    continue
                matched_words += 1
                ref_time = float(ref_timed[ref_idx]["time"])
                hyp_time = float(hyp_timed[hyp_idx]["time"])
                excess_lateness = hyp_time - ref_time - float(late_word_tolerance_seconds)
                if excess_lateness > 0:
                    late_count += 1
                    late_penalty += min(
                        float(late_word_penalty_max),
                        excess_lateness * float(late_word_penalty_per_second),
                    )
            if late_count > 0:
                late_correct_words += late_count
                late_correct_penalty += late_penalty
                wer += late_penalty / ref_count
        cer, *_ = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=True)
        error = wer_weight * float(wer) + cer_weight * float(cer)
        reward = reward_offset - reward_scale * error
        if reward_min is not None:
            reward = max(float(reward_min), reward)
        if reward_max is not None:
            reward = min(float(reward_max), reward)
        if reward_positive_threshold is not None and reward <= reward_positive_threshold:
            reward = 0.0
        rewards.append(reward)
    stats = {
        "late_correct_words": float(late_correct_words),
        "late_correct_penalty": float(late_correct_penalty),
        "matched_words": float(matched_words),
        "late_correct_word_fraction": float(late_correct_words) / max(float(matched_words), 1.0),
        "late_correct_penalty_fraction": float(late_correct_penalty) / max(float(matched_words), 1.0),
    }
    return torch.tensor(rewards, dtype=torch.float32), stats


def _optional_float(config: Any, key: str, default: Optional[float]) -> Optional[float]:
    value = config.get(key, default)
    if value is None:
        return None
    return float(value)


def compute_rewards(
    hypotheses: List[str],
    references: List[str],
    reward_config: Any,
    hypothesis_word_times: Optional[List[List[Dict[str, Any]]]] = None,
    reference_word_times: Optional[List[List[Dict[str, Any]]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    reward_type = reward_config.get("reward_type", "weighted_error")
    if reward_type != "weighted_error":
        raise ValueError(f"unknown streaming RL reward_type {reward_type}")
    return weighted_error_rewards(
        hypotheses=hypotheses,
        references=references,
        hypothesis_word_times=hypothesis_word_times,
        reference_word_times=reference_word_times,
        late_word_tolerance_seconds=_optional_float(reward_config, "late_word_tolerance_seconds", None),
        late_word_penalty_per_second=float(reward_config.get("late_word_penalty_per_second", 0.25)),
        late_word_penalty_max=float(reward_config.get("late_word_penalty_max", 1.0)),
        wer_weight=float(reward_config.get("reward_wer_weight", 0.7)),
        cer_weight=float(reward_config.get("reward_cer_weight", 0.3)),
        reward_offset=float(reward_config.get("reward_offset", 1.0)),
        reward_scale=float(reward_config.get("reward_scale", 1.0)),
        reward_min=_optional_float(reward_config, "reward_min", 0.0),
        reward_max=_optional_float(reward_config, "reward_max", None),
        reward_positive_threshold=_optional_float(reward_config, "reward_positive_threshold", None),
    )


def compute_grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float,
    min_group_std: float = 0.0,
) -> torch.Tensor:
    grouped = rearrange(rewards, "(b g) -> b g", g=group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    active = std > max(eps, min_group_std)
    advantages = (grouped - mean) / std.clamp_min(eps)
    advantages = torch.where(active, advantages, torch.zeros_like(advantages))
    return rearrange(advantages, "b g -> (b g)")


def streaming_reference_for_reward(
    transcript: Sequence[Dict[str, Any]],
    chunk_start_frames: int,
    output_length: int,
    subsampling_factor: int,
    delay_seconds: float,
    normalizer: Any,
) -> str:
    del chunk_start_frames, output_length, subsampling_factor, delay_seconds
    return reference_words(resolve_timed_words(transcript), normalizer=normalizer)


def make_streaming_rl_chunks(
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    transcripts: Sequence[Any],
    chunk_size: int,
    chunk_overlap: int,
    delay_seconds: float,
    buffer_seconds: float,
    include_empty_references: bool,
) -> List[Dict[str, Any]]:
    stride = chunk_size - chunk_overlap
    if stride <= 0:
        raise ValueError("audio_chunking.size must be greater than overlap")

    chunks = []
    for chunk_start in range(0, int(audio_lengths.max().item()), stride):
        active = audio_lengths > chunk_start
        if active.sum().item() == 0:
            continue

        chunk = audio[active, :, chunk_start : chunk_start + chunk_size]
        chunk_lengths = torch.clamp(audio_lengths[active] - chunk_start, min=0, max=chunk.size(-1))
        chunk_transcripts = [
            filter_words_by_frame_overlap(transcripts[i], chunk_start, chunk_start + chunk_size)
            for i, keep in enumerate(active.tolist())
            if keep
        ]
        if not include_empty_references:
            non_empty = [idx for idx, transcript in enumerate(chunk_transcripts) if len(resolve_timed_words(transcript)) > 0]
            if len(non_empty) == 0:
                continue
            non_empty_t = torch.tensor(non_empty, dtype=torch.long)
            chunk = chunk.index_select(0, non_empty_t)
            chunk_lengths = chunk_lengths.index_select(0, non_empty_t)
            chunk_transcripts = [chunk_transcripts[idx] for idx in non_empty]

        chunk, chunk_lengths = pad_audio_for_streaming_delay(
            chunk,
            chunk_lengths,
            delay_seconds=delay_seconds,
            buffer_seconds=buffer_seconds,
        )
        chunks.append(
            {
                "audio": chunk,
                "audio_lengths": chunk_lengths,
                "transcripts": chunk_transcripts,
                "chunk_start_frames": torch.full((len(chunk_transcripts),), chunk_start, dtype=torch.long),
            }
        )
    return chunks


def _layer_supports_kv_cache(layer: torch.nn.Module) -> bool:
    attn = getattr(layer, "attn", None)
    return (
        attn is not None
        and hasattr(layer, "attn_norm")
        and hasattr(layer, "ff_norm")
        and hasattr(layer, "ff")
        and hasattr(attn, "qkv")
        and hasattr(attn, "out_proj")
        and hasattr(attn, "apply_rotary")
    )


def rollout_kv_cache_enabled(model: torch.nn.Module) -> bool:
    return all(_layer_supports_kv_cache(layer) for layer in model.layers)


def _rotary_emb_fn_for_step(model: torch.nn.Module, step: int, device: torch.device):
    rotary_pos_emb = getattr(model, "rotary_pos_emb", None)
    if rotary_pos_emb is None:
        return None
    cos, sin = rotary_pos_emb(step + 1, device)
    return apply_rotary(
        cos=cos[:, step : step + 1],
        sin=sin[:, step : step + 1],
        learned=rotary_pos_emb.learned_freq,
    )


def _cached_decoder_layer_step(
    layer: torch.nn.Module,
    x_step: torch.Tensor,
    cached_kv: Optional[torch.Tensor],
    rotary_emb_fn: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attn = layer.attn
    x_norm = layer.attn_norm(x_step)
    q, k, v = attn.qkv(x_norm)
    current_kv = torch.stack([k, v], dim=2)
    q, current_kv = attn.apply_rotary(q, current_kv, rotary_emb_fn)
    full_kv = current_kv if cached_kv is None else torch.cat([cached_kv, current_kv], dim=1)

    q = q.transpose(1, 2).contiguous()
    k, v = rearrange(full_kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
    )
    attn_out = rearrange(attn_out, "b h n d -> b n (h d)")
    x_step = x_step + attn.out_proj(attn_out)
    x_step = x_step + layer.ff(layer.ff_norm(x_step))
    return x_step, full_kv


@torch.no_grad()
def sample_streaming_rollouts(
    model: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    num_rollouts: int,
    temperature: float,
    max_output_frames: Optional[int] = None,
    use_kv_cache: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    was_training = model.training
    model.eval()

    x = audio.transpose(1, 2)
    x, out_lengths = model.subsampling(x, lengths=audio_lengths)
    if max_output_frames is not None and max_output_frames > 0 and x.size(1) > max_output_frames:
        x = x[:, :max_output_frames]
        out_lengths = out_lengths.clamp(max=max_output_frames)

    x = x.repeat_interleave(num_rollouts, dim=0)
    out_lengths = out_lengths.repeat_interleave(num_rollouts, dim=0)
    key_padding_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) >= out_lengths.unsqueeze(1)
    key_padding_mask = key_padding_mask if key_padding_mask.any() else None

    prev_ids = torch.full((x.size(0), x.size(1)), model.get_silence_id(), dtype=torch.long, device=x.device)
    predictions = []
    temperature = max(float(temperature), 1e-6)
    detected_kv_cache = rollout_kv_cache_enabled(model)
    if use_kv_cache is None:
        use_kv_cache = detected_kv_cache
    elif use_kv_cache and not detected_kv_cache:
        raise ValueError("requested rollout KV cache, but model layers do not expose the required cacheable structure")
    layer_kv_cache: List[Optional[torch.Tensor]] = [None for _ in model.layers]

    for step in range(x.size(1)):
        if use_kv_cache:
            h = x[:, step : step + 1] + model.prev_token_embedding(prev_ids[:, step : step + 1])
            valid = (step < out_lengths).view(-1, 1, 1)
            h = h.masked_fill(~valid, 0)
            rotary_emb_fn = _rotary_emb_fn_for_step(model, step, h.device)
            for layer_idx, layer in enumerate(model.layers):
                h, layer_kv_cache[layer_idx] = _cached_decoder_layer_step(
                    layer=layer,
                    x_step=h,
                    cached_kv=layer_kv_cache[layer_idx],
                    rotary_emb_fn=rotary_emb_fn,
                )
            step_h = model.norm(h.squeeze(1))
        else:
            h = x + model.prev_token_embedding(prev_ids)
            if key_padding_mask is not None:
                h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0)
            rotary_emb_fn = model._rotary_emb_fn(h.size(1), h.device) if hasattr(model, "_rotary_emb_fn") else None
            for layer in model.layers:
                h = layer(h, rotary_emb_fn=rotary_emb_fn)
            step_h = model.norm(h[:, step])
        logits = model._combined_logits(model.silence_head(step_h), model.text_head(step_h))
        pred = torch.multinomial((logits / temperature).softmax(dim=-1), num_samples=1).squeeze(-1)
        valid = step < out_lengths
        pred = torch.where(valid, pred, torch.full_like(pred, model.get_silence_id()))
        predictions.append(pred)
        if step + 1 < x.size(1):
            prev_ids[:, step + 1] = pred

    if was_training:
        model.train()
    return torch.stack(predictions, dim=1), out_lengths


def streaming_sequence_logprobs(
    model: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    actions: torch.Tensor,
    action_lengths: torch.Tensor,
    num_rollouts: int,
) -> torch.Tensor:
    repeated_audio = audio.repeat_interleave(num_rollouts, dim=0)
    repeated_lengths = audio_lengths.repeat_interleave(num_rollouts, dim=0)
    tensor_lengths = torch.full((audio.size(0),), audio.size(-1), dtype=audio_lengths.dtype, device=audio_lengths.device)
    full_output_length = int(model.output_lengths(tensor_lengths).max().item())
    if actions.size(1) < full_output_length:
        pad = full_output_length - actions.size(1)
        actions_for_forward = torch.nn.functional.pad(actions, (0, pad), value=model.get_silence_id())
    else:
        actions_for_forward = actions
    out = model(audio_signal=repeated_audio, length=repeated_lengths, frame_targets=actions_for_forward, return_logits=True)
    logits = out["logits"]
    if logits.size(1) != actions.size(1):
        common = min(logits.size(1), actions.size(1))
        logits = logits[:, :common]
        actions = actions[:, :common]
        action_lengths = action_lengths.clamp(max=common)
    log_probs = logits.log_softmax(dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    mask = torch.arange(actions.size(1), device=actions.device).expand(actions.size(0), -1) < action_lengths.unsqueeze(1)
    return token_log_probs.masked_fill(~mask, 0.0).sum(dim=1)


def slice_streaming_chunk(chunk: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    return {
        "audio": chunk["audio"][start:end],
        "audio_lengths": chunk["audio_lengths"][start:end],
        "transcripts": chunk["transcripts"][start:end],
        "chunk_start_frames": chunk["chunk_start_frames"][start:end],
    }


def aggregate_microbatch_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(metrics_list) == 1:
        return metrics_list[0]

    total_base = sum(int(metrics["chunk_batch"]) for metrics in metrics_list)

    def weighted_mean(key: str) -> float:
        return sum(float(metrics[key]) * int(metrics["chunk_batch"]) for metrics in metrics_list) / max(total_base, 1)

    late_correct_words = sum(float(metrics["late_correct_words"]) for metrics in metrics_list)
    late_correct_penalty = sum(float(metrics["late_correct_penalty"]) for metrics in metrics_list)
    matched_words = sum(float(metrics["matched_words"]) for metrics in metrics_list)
    output = {
        "loss": weighted_mean("loss"),
        "reward_mean": weighted_mean("reward_mean"),
        "reward_max": max(float(metrics["reward_max"]) for metrics in metrics_list),
        "reward_min": min(float(metrics["reward_min"]) for metrics in metrics_list),
        "reward_group_std_mean": weighted_mean("reward_group_std_mean"),
        "active_reward_group_fraction": weighted_mean("active_reward_group_fraction"),
        "skipped_low_reward_std": weighted_mean("skipped_low_reward_std"),
        "advantage_abs_mean": weighted_mean("advantage_abs_mean"),
        "zero_advantage": all(bool(metrics["zero_advantage"]) for metrics in metrics_list),
        "sample_reward": metrics_list[0]["sample_reward"],
        "sample_hypothesis": metrics_list[0]["sample_hypothesis"],
        "sample_reference": metrics_list[0]["sample_reference"],
        "late_correct_words": late_correct_words,
        "late_correct_penalty": late_correct_penalty,
        "matched_words": matched_words,
        "late_correct_word_fraction": late_correct_words / max(matched_words, 1.0),
        "late_correct_penalty_fraction": late_correct_penalty / max(matched_words, 1.0),
        "output_frames": max(int(metrics["output_frames"]) for metrics in metrics_list),
        "chunk_batch": total_base,
        "microbatches": len(metrics_list),
    }
    return output


def rl_loss_for_chunk(
    model: torch.nn.Module,
    chunk: Dict[str, Any],
    tokenizer: Any,
    config: OmegaConf,
    device: torch.device,
    dtype: torch.dtype,
    normalizer: Any,
) -> Tuple[torch.Tensor, bool, Dict[str, Any]]:
    rl_config = config.rl
    num_rollouts = int(rl_config.num_rollouts)
    model_dtype = next(model.parameters()).dtype

    audio = chunk["audio"].to(device=device, dtype=model_dtype)
    audio_lengths = chunk["audio_lengths"].to(device)
    chunk_start_frames = chunk["chunk_start_frames"]
    transcripts = chunk["transcripts"]
    max_output_frames = rl_config.get("max_output_frames", None)
    max_output_frames = None if max_output_frames is None else int(max_output_frames)

    with torch.autocast(device.type, dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext():
        actions, action_lengths = sample_streaming_rollouts(
            model=model,
            audio=audio,
            audio_lengths=audio_lengths,
            num_rollouts=num_rollouts,
            temperature=float(rl_config.temperature),
            max_output_frames=max_output_frames,
        )

        hypotheses = []
        hypothesis_word_times = []
        for row_idx in range(actions.size(0)):
            prediction_ids = actions[row_idx, : int(action_lengths[row_idx].item())].detach().cpu().tolist()
            base_idx = row_idx // num_rollouts
            hypotheses.append(
                decode_model_prediction_ids(
                    model=model,
                    tokenizer=tokenizer,
                    prediction_ids=prediction_ids,
                    max_tokens=None,
                    normalizer=normalizer,
                )
            )
            hypothesis_word_times.append(
                prediction_words_with_times(
                    model=model,
                    tokenizer=tokenizer,
                    prediction_ids=prediction_ids,
                    chunk_start_frames=int(chunk_start_frames[base_idx].item()),
                    subsampling_factor=model.subsampling_factor,
                    normalizer=normalizer,
                )
            )
        base_lengths = action_lengths.view(-1, num_rollouts)[:, 0].detach().cpu()
        references = []
        reference_word_times = []
        for batch_idx, transcript in enumerate(transcripts):
            reference = streaming_reference_for_reward(
                transcript=transcript,
                chunk_start_frames=int(chunk_start_frames[batch_idx].item()),
                output_length=int(base_lengths[batch_idx].item()),
                subsampling_factor=model.subsampling_factor,
                delay_seconds=float(config.streaming.get("delay_seconds", 2.0)),
                normalizer=normalizer,
            )
            references.extend([reference] * num_rollouts)
            ref_timed = timed_reference_words(transcript, normalizer=normalizer)
            reference_word_times.extend([ref_timed] * num_rollouts)

        rewards, reward_stats = compute_rewards(
            hypotheses=hypotheses,
            references=references,
            reward_config=rl_config,
            hypothesis_word_times=hypothesis_word_times,
            reference_word_times=reference_word_times,
        )
        rewards = rewards.to(device)
        advantages = compute_grpo_advantages(
            rewards=rewards,
            group_size=num_rollouts,
            eps=float(rl_config.get("advantage_eps", 1e-6)),
            min_group_std=float(rl_config.get("reward_std_min", 0.0)),
        ).to(device)

        if advantages.abs().sum() == 0:
            loss = torch.zeros((), device=device, requires_grad=True)
            skipped_zero_advantage = True
        else:
            was_training = model.training
            model.eval()
            try:
                logprobs = streaming_sequence_logprobs(
                    model=model,
                    audio=audio,
                    audio_lengths=audio_lengths,
                    actions=actions,
                    action_lengths=action_lengths,
                    num_rollouts=num_rollouts,
                )
            finally:
                if was_training:
                    model.train()
            loss = -(advantages.detach() * logprobs).mean()
            skipped_zero_advantage = False

    rewards_grouped = rearrange(rewards.detach().cpu(), "(b g) -> b g", g=num_rollouts)
    reward_std_grouped = rewards_grouped.std(dim=1, unbiased=False)
    reward_std_min = float(rl_config.get("reward_std_min", 0.0))
    active_groups = reward_std_grouped > max(float(rl_config.get("advantage_eps", 1e-6)), reward_std_min)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "reward_mean": float(rewards.mean().detach().cpu()),
        "reward_max": float(rewards.max().detach().cpu()),
        "reward_min": float(rewards.min().detach().cpu()),
        "reward_group_std_mean": float(reward_std_grouped.mean().item()),
        "active_reward_group_fraction": float(active_groups.float().mean().item()),
        "skipped_low_reward_std": float((~active_groups).float().mean().item()),
        "advantage_abs_mean": float(advantages.abs().mean().detach().cpu()),
        "zero_advantage": skipped_zero_advantage,
        "sample_reward": float(rewards[0].detach().cpu()) if len(rewards) > 0 else 0.0,
        "sample_hypothesis": hypotheses[0] if len(hypotheses) > 0 else "",
        "sample_reference": references[0] if len(references) > 0 else "",
        "late_correct_words": reward_stats["late_correct_words"],
        "late_correct_penalty": reward_stats["late_correct_penalty"],
        "matched_words": reward_stats["matched_words"],
        "late_correct_word_fraction": reward_stats["late_correct_word_fraction"],
        "late_correct_penalty_fraction": reward_stats["late_correct_penalty_fraction"],
        "output_frames": int(action_lengths.max().detach().cpu().item()),
        "chunk_batch": len(transcripts),
        "microbatches": 1,
    }
    return loss, skipped_zero_advantage, metrics


def rl_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    chunk: Dict[str, Any],
    tokenizer: Any,
    config: OmegaConf,
    device: torch.device,
    dtype: torch.dtype,
    normalizer: Any,
) -> Dict[str, Any]:
    batch_size = len(chunk["transcripts"])
    microbatch_size = int(config.rl.get("microbatch_size", 0) or 0)
    if microbatch_size <= 0:
        microbatch_size = batch_size
    microbatch_size = max(1, min(microbatch_size, batch_size))

    optimizer.zero_grad()
    metrics_list = []
    any_update = False
    for start in range(0, batch_size, microbatch_size):
        micro_chunk = slice_streaming_chunk(chunk, start, min(start + microbatch_size, batch_size))
        loss, skipped_zero_advantage, metrics = rl_loss_for_chunk(
            model=model,
            chunk=micro_chunk,
            tokenizer=tokenizer,
            config=config,
            device=device,
            dtype=dtype,
            normalizer=normalizer,
        )
        metrics_list.append(metrics)
        if not skipped_zero_advantage:
            loss_scale = float(metrics["chunk_batch"]) / float(batch_size)
            (loss * loss_scale).backward()
            any_update = True
        del loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any_update:
        clip_value = float(config.training.get("clip_value", 0.8))
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

    metrics = aggregate_microbatch_metrics(metrics_list)
    metrics["microbatch_size"] = microbatch_size
    return metrics


def init_wandb(config: OmegaConf, config_path: str):
    if not config.wandb.get("use", False):
        return None
    wandb_dir = config.wandb.get("dir", "./wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    run_id = config.wandb.get("id", "")
    kwargs = {
        "project": config.wandb.project_name,
        "name": config.wandb.get("name", None),
        "config": OmegaConf.to_container(config, resolve=True),
        "dir": wandb_dir,
    }
    if run_id:
        run = wandb.init(id=run_id, resume="must", allow_val_change=True, **kwargs)
    else:
        run = wandb.init(**kwargs)
    config.wandb.id = wandb.run.id
    if config.wandb.get("update_config_with_wandb_id", False):
        OmegaConf.save(config=config, f=config_path)
    print(f"\nLogging with WandB id: {wandb.run.id}\n")
    return run


def make_dataloader(config: OmegaConf, tokenizer: Any, args: argparse.Namespace, seen_ids: List[str], epoch: int):
    paired_data = lcasr.utils.audio_tools.load_json(config.data.path)
    max_records = config.data.get("max_records", None)
    if max_records is not None:
        paired_data = dict(list(paired_data.items())[: int(max_records)])
    seed = int(config.training.get("random_seed", 1234))
    return VariableBatchSimpleDataloader(
        pairs=paired_data,
        tokenizer=tokenizer,
        batch_size=int(config.training.batch_size),
        chunk_size=int(config.audio_chunking.size),
        chunk_overlap=int(config.audio_chunking.get("overlap", 0)),
        num_workers=int(config.training.get("num_workers", args.num_workers)),
        pin_memory=bool(config.training.get("pin_memory", args.pin_memory)),
        prefetch=config.training.get("prefetch_factor", args.prefetch_factor),
        seen_ids=seen_ids,
        random_seed=seed + epoch,
    )


def scheduler_step(scheduler: torch.optim.lr_scheduler._LRScheduler, step: int, max_steps: int) -> None:
    if scheduler is None:
        return
    if hasattr(scheduler, "is_warmup"):
        was_warmup = scheduler.is_warmup
        if was_warmup:
            scheduler.is_warmup = scheduler.is_warming_up()
            if not scheduler.is_warmup and was_warmup:
                scheduler.set_cosine_schedule(total_recordings=max_steps, cur_podcast=step)
        if scheduler.is_warmup:
            scheduler.step()
        else:
            scheduler.step(epoch=step)
    else:
        scheduler.step()


def wandb_scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, bool)) and not isinstance(value, str)
    }


def wandb_rollout_sample_table(metrics: Dict[str, Any], step: int) -> wandb.Table:
    return wandb.Table(
        columns=["step", "reward", "hypothesis", "reference"],
        data=[
            [
                step,
                metrics.get("sample_reward", 0.0),
                metrics.get("sample_hypothesis", ""),
                metrics.get("sample_reference", ""),
            ]
        ],
    )


def state_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_to_cpu(item) for item in value)
    return value


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: OmegaConf,
    seen_ids: List[str],
    epoch: int,
    other: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(config.checkpointing.dir, exist_ok=True)
    save_path = os.path.join(config.checkpointing.dir, f"step_{step}.pt")
    print(f"checkpoint_save_start step={step} path={save_path}", flush=True)
    checkpoint = {
        "model": state_to_cpu(model.state_dict()),
        "optimizer": state_to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": state_to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "podcast_step": step,
        "config": config,
        "seen_ids": seen_ids,
        "epoch": epoch,
    }
    if other:
        checkpoint.update(state_to_cpu(other))
    print(f"checkpoint_torch_save_start step={step}", flush=True)
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)
    print(f"checkpoint_save_done step={step}", flush=True)


def apply_cli_overrides(config: OmegaConf, args: argparse.Namespace) -> OmegaConf:
    if args.checkpoint_dir is not None:
        config.checkpointing.dir = args.checkpoint_dir
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.data_path is not None:
        config.data.path = args.data_path
    if args.max_records is not None:
        config.data.max_records = args.max_records
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.disable_wandb:
        config.wandb.use = False
    return config


def train(args: argparse.Namespace) -> None:
    args.config_path = args.config
    config = apply_cli_overrides(OmegaConf.load(args.config), args)
    if config.rl.get("algorithm", "grpo") != "grpo":
        raise ValueError("streaming decoder RL currently supports rl.algorithm: grpo")
    os.makedirs(config.checkpointing.dir, exist_ok=True)

    tokenizer_kwargs = {}
    if "tokenizer_path" in config.training:
        tokenizer_kwargs["tokenizer_path"] = config.training.tokenizer_path
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)

    seed = config.training.get("random_seed", 1234)
    if seed == "random":
        seed = int(time.time()) % 10000
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
    total_params = model.print_total_params()
    model = model.to(device)
    optimizer, scheduler = load_optimizer(config, model)
    seen_ids, step, epoch = load_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        path=config.checkpointing.dir,
        device=device,
    )
    if args.reset_step:
        seen_ids, step, epoch = [], 0, 0
    run = init_wandb(config, args.config_path)

    dtype = get_dtype(config.training.get("dtype", "bfloat16"))
    max_steps = int(config.training.max_steps)
    chunk_size = int(config.audio_chunking.size)
    chunk_overlap = int(config.audio_chunking.get("overlap", 0))
    delay_seconds = float(config.streaming.get("delay_seconds", 2.0))
    buffer_seconds = float(config.streaming.get("buffer_seconds", 0.25))
    include_empty_references = bool(config.rl.get("include_empty_references", False))
    normalizer = EnglishTextNormalizer() if EnglishTextNormalizer is not None else None
    last_saved_step = None

    print(f"Streaming decoder RL params: {total_params / 1e6:.2f}M")
    print(f"Starting from step: {step}")
    print(f"RL algorithm: {config.rl.algorithm}")
    print(f"Rollouts per chunk: {config.rl.num_rollouts}")
    print(f"RL microbatch size: {config.rl.get('microbatch_size', 0)}")
    print(f"Rollout KV cache: {rollout_kv_cache_enabled(model)}")
    print(f"Reward std minimum: {config.rl.get('reward_std_min', 0.0)}")
    print(f"Max output frames: {config.rl.get('max_output_frames', None)}")
    print(f"Late word tolerance seconds: {config.rl.get('late_word_tolerance_seconds', None)}")
    print(f"Late word penalty per second: {config.rl.get('late_word_penalty_per_second', 0.25)}")
    print(f"Late word penalty max: {config.rl.get('late_word_penalty_max', 1.0)}")

    pbar = tqdm(total=max_steps, initial=step, desc="Streaming decoder RL updates")
    while step < max_steps:
        dataloader = make_dataloader(config=config, tokenizer=tokenizer, args=args, seen_ids=seen_ids, epoch=epoch)
        for batch in dataloader:
            audio, audio_lengths, transcripts, ids = batch
            seen_ids.extend(ids)
            chunks = make_streaming_rl_chunks(
                audio=audio,
                audio_lengths=audio_lengths,
                transcripts=transcripts,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                delay_seconds=delay_seconds,
                buffer_seconds=buffer_seconds,
                include_empty_references=include_empty_references,
            )
            if bool(config.rl.get("shuffle_chunks", True)):
                random.shuffle(chunks)

            for chunk in chunks:
                if step >= max_steps:
                    break
                metrics = rl_update(
                    model=model,
                    optimizer=optimizer,
                    chunk=chunk,
                    tokenizer=tokenizer,
                    config=config,
                    device=device,
                    dtype=dtype,
                    normalizer=normalizer,
                )
                step += 1
                scheduler_step(scheduler, step=step, max_steps=max_steps)
                metrics["learning_rate"] = scheduler.get_last_lr()[0] if scheduler is not None else config.optimizer.args.lr
                metrics["step"] = step
                metrics["epoch"] = epoch
                pbar.update(1)
                pbar.set_postfix(
                    reward=f"{metrics['reward_mean']:.3f}",
                    active=f"{metrics['active_reward_group_fraction']:.3f}",
                    loss=f"{metrics['loss']:.3f}",
                )
                if run is not None:
                    wandb_payload = wandb_scalar_metrics(metrics)
                    text_log_every = int(config.rl.get("sample_text_log_every", 0))
                    if text_log_every > 0 and step % text_log_every == 0:
                        wandb_payload["rollout_sample"] = wandb_rollout_sample_table(metrics, step=step)
                    wandb.log(wandb_payload, step=step)

                if step % int(config.checkpointing.save_every_n_steps) == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        config=config,
                        seen_ids=seen_ids,
                        epoch=epoch,
                        other={"rl_metrics": metrics},
                    )
                    last_saved_step = step
            if step >= max_steps:
                break
        epoch += 1
        seen_ids = reset_seen_ids(seen_ids=seen_ids, epoch=epoch - 1)

    if last_saved_step != step:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=config,
            seen_ids=seen_ids,
            epoch=epoch,
        )
    if run is not None:
        wandb.finish()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def validate_config(args: argparse.Namespace) -> None:
    config = apply_cli_overrides(OmegaConf.load(args.config), args)
    assert config.rl.algorithm == "grpo"
    assert int(config.rl.num_rollouts) > 1
    assert int(config.training.max_steps) > 0
    assert os.path.exists(config.data.path), f"data path missing: {config.data.path}"
    os.makedirs(config.checkpointing.dir, exist_ok=True)
    if config.wandb.get("use", False):
        os.makedirs(config.wandb.get("dir", "./wandb"), exist_ok=True)
    if args.validate_load_model:
        tokenizer_kwargs = {}
        if "tokenizer_path" in config.training:
            tokenizer_kwargs["tokenizer_path"] = config.training.tokenizer_path
        tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)
        model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
        incompatible = load_checkpoint(
            args=args,
            model=model,
            path=config.checkpointing.dir,
            device="cpu",
        )
        print(f"Checkpoint load validation passed: {incompatible[1]} starting step")
    print("Config validation passed")


def smoke_rollout(args: argparse.Namespace) -> None:
    config = apply_cli_overrides(OmegaConf.load(args.config), args)
    config.wandb.use = False
    config.rl.num_rollouts = int(args.smoke_num_rollouts)
    if args.smoke_max_output_frames is not None:
        config.rl.max_output_frames = int(args.smoke_max_output_frames)
    config.training.max_steps = 1
    config.data.max_records = int(args.smoke_max_records)

    tokenizer_kwargs = {}
    if "tokenizer_path" in config.training:
        tokenizer_kwargs["tokenizer_path"] = config.training.tokenizer_path
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_cpu else "cpu")

    model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
    model = model.to(device)
    print(f"Smoke rollout KV cache: {rollout_kv_cache_enabled(model)}")
    optimizer, _ = load_optimizer(config, model)
    load_checkpoint(args=args, model=model, optimizer=None, path=config.checkpointing.dir, device=device)
    normalizer = EnglishTextNormalizer() if EnglishTextNormalizer is not None else None
    dataloader = make_dataloader(config=config, tokenizer=tokenizer, args=args, seen_ids=[], epoch=0)
    audio, audio_lengths, transcripts, _ = next(iter(dataloader))
    chunks = make_streaming_rl_chunks(
        audio=audio,
        audio_lengths=audio_lengths,
        transcripts=transcripts,
        chunk_size=int(config.audio_chunking.size),
        chunk_overlap=int(config.audio_chunking.get("overlap", 0)),
        delay_seconds=float(config.streaming.get("delay_seconds", 2.0)),
        buffer_seconds=float(config.streaming.get("buffer_seconds", 0.25)),
        include_empty_references=bool(config.rl.get("include_empty_references", False)),
    )
    if len(chunks) == 0:
        raise RuntimeError("no non-empty streaming RL chunks produced for smoke batch")
    metrics = rl_update(
        model=model,
        optimizer=optimizer,
        chunk=chunks[0],
        tokenizer=tokenizer,
        config=config,
        device=device,
        dtype=get_dtype(config.training.get("dtype", "bfloat16")),
        normalizer=normalizer,
    )
    print(
        "Smoke rollout passed "
        f"(device={device}, chunk_batch={chunks[0]['audio'].shape[0]}, rollouts={config.rl.num_rollouts}, "
        f"output_frames={metrics['output_frames']}, reward_mean={metrics['reward_mean']:.4f}, "
        f"active_groups={metrics['active_reward_group_fraction']:.4f}, loss={metrics['loss']:.4f})"
    )


def self_test() -> None:
    overlap_words = filter_words_by_frame_overlap(
        [
            {"start": 0.5, "end": 0.9, "text": "before"},
            {"start": 0.9, "end": 1.1, "text": "left"},
            {"start": 1.5, "end": 2.2, "text": "right"},
            {"start": 2.1, "end": 2.3, "text": "after"},
        ],
        start_frame=100,
        end_frame=200,
    )
    assert [word["text"] for word in overlap_words] == ["left", "right"]
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.9, 0.1, 0.5])
    advantages = compute_grpo_advantages(rewards, group_size=3, eps=1e-6, min_group_std=0.02)
    assert advantages[:3].abs().sum() == 0
    assert advantages[3:].abs().sum() > 0
    reward, reward_stats = weighted_error_rewards(["hello world", "hello"], ["hello world", "hello world"])
    assert reward[0].item() == 1.0
    assert reward[1].item() < 1.0
    assert reward_stats["late_correct_words"] == 0.0
    late_reward, late_stats = weighted_error_rewards(
        ["hello world"],
        ["hello world"],
        hypothesis_word_times=[[{"word": "hello", "time": 0.5}, {"word": "world", "time": 5.5}]],
        reference_word_times=[[{"word": "hello", "time": 0.4}, {"word": "world", "time": 1.0}]],
        late_word_tolerance_seconds=2.0,
    )
    assert late_stats["late_correct_words"] == 1.0
    assert late_stats["late_correct_penalty"] == 0.625
    assert late_reward[0].item() < 1.0
    mild_late_reward, mild_late_stats = weighted_error_rewards(
        ["world"],
        ["world"],
        hypothesis_word_times=[[{"word": "world", "time": 4.0}]],
        reference_word_times=[[{"word": "world", "time": 1.0}]],
        late_word_tolerance_seconds=2.0,
        late_word_penalty_per_second=0.25,
        late_word_penalty_max=1.0,
    )
    capped_late_reward, capped_late_stats = weighted_error_rewards(
        ["world"],
        ["world"],
        hypothesis_word_times=[[{"word": "world", "time": 8.0}]],
        reference_word_times=[[{"word": "world", "time": 1.0}]],
        late_word_tolerance_seconds=2.0,
        late_word_penalty_per_second=0.25,
        late_word_penalty_max=1.0,
    )
    assert mild_late_stats["late_correct_penalty"] == 0.25
    assert capped_late_stats["late_correct_penalty"] == 1.0
    assert capped_late_reward[0].item() < mild_late_reward[0].item()
    on_time_reward, on_time_stats = weighted_error_rewards(
        ["hello world"],
        ["hello world"],
        hypothesis_word_times=[[{"word": "hello", "time": 0.5}, {"word": "world", "time": 2.9}]],
        reference_word_times=[[{"word": "hello", "time": 0.4}, {"word": "world", "time": 1.0}]],
        late_word_tolerance_seconds=2.0,
    )
    assert on_time_stats["late_correct_words"] == 0.0
    assert on_time_reward[0].item() == 1.0
    repeated_late_reward, repeated_late_stats = weighted_error_rewards(
        ["go go now"],
        ["go go now"],
        hypothesis_word_times=[
            [{"word": "go", "time": 0.3}, {"word": "go", "time": 4.2}, {"word": "now", "time": 4.4}]
        ],
        reference_word_times=[
            [{"word": "go", "time": 0.2}, {"word": "go", "time": 1.0}, {"word": "now", "time": 4.0}]
        ],
        late_word_tolerance_seconds=2.0,
    )
    assert repeated_late_stats["late_correct_words"] == 1.0
    assert repeated_late_reward[0].item() < 1.0
    substitution_late_reward, substitution_late_stats = weighted_error_rewards(
        ["hello wurld"],
        ["hello world"],
        hypothesis_word_times=[[{"word": "hello", "time": 0.5}, {"word": "wurld", "time": 6.0}]],
        reference_word_times=[[{"word": "hello", "time": 0.4}, {"word": "world", "time": 1.0}]],
        late_word_tolerance_seconds=2.0,
    )
    assert substitution_late_stats["late_correct_words"] == 0.0
    assert substitution_late_reward[0].item() < 1.0
    table = wandb_rollout_sample_table(
        {"sample_reward": 0.5, "sample_hypothesis": "hello", "sample_reference": "hello world"},
        step=7,
    )
    assert len(table.data) == 1

    class ToyTokenizer:
        def vocab_size(self):
            return 4

        def encode(self, text):
            return [1 for token in text.split() if token]

        def decode(self, tokens):
            return " ".join(f"tok{int(token)}" for token in tokens)

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_rotary = False

        def forward(self, x, rotary_emb_fn=None):
            self.seen_rotary = self.seen_rotary or rotary_emb_fn is not None
            return x

    class ToyStreaming(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.silence_id = 4
            self.subsampling_factor = 1
            self.prev_token_embedding = torch.nn.Embedding(5, 3)
            self.proj = torch.nn.Linear(3, 5)
            self.layers = torch.nn.ModuleList([ToyLayer()])
            self.norm = torch.nn.Identity()

        def get_silence_id(self):
            return self.silence_id

        def _decode_prediction_ids(self, tokenizer, prediction_ids, max_tokens=None):
            tokens = []
            for idx in prediction_ids:
                idx = int(idx)
                if idx == self.silence_id:
                    continue
                tokens.append(idx)
                if max_tokens is not None and len(tokens) >= max_tokens:
                    break
            return "" if not tokens else tokenizer.decode(tokens)

        def output_lengths(self, lengths):
            return lengths

        def subsampling(self, x, lengths):
            return torch.zeros(x.size(0), x.size(1), 3), lengths

        def _rotary_emb_fn(self, length, device):
            del length, device
            return lambda q, k: (q, k)

        def _combined_logits(self, silence_logits, text_logits):
            return self.proj(torch.zeros(text_logits.size(0), 3))

        def silence_head(self, h):
            return torch.zeros(h.size(0), 2)

        def text_head(self, h):
            return torch.zeros(h.size(0), 4)

        def forward(self, audio_signal, length, frame_targets, return_logits=True):
            h = self.prev_token_embedding(frame_targets.clamp_min(0))
            return {"logits": self.proj(h), "length": length}

    toy = ToyStreaming()
    repeated_text = decode_model_prediction_ids(
        model=toy,
        tokenizer=ToyTokenizer(),
        prediction_ids=[1, 1, toy.get_silence_id(), toy.get_silence_id(), 2, 2],
        max_tokens=None,
        normalizer=None,
    )
    assert repeated_text == "tok1 tok1 tok2 tok2"
    audio = torch.zeros(2, 80, 4)
    lengths = torch.tensor([4, 3], dtype=torch.long)
    actions, action_lengths = sample_streaming_rollouts(toy, audio, lengths, num_rollouts=2, temperature=1.0)
    assert actions.shape == (4, 4)
    assert toy.layers[0].seen_rotary
    logprobs = streaming_sequence_logprobs(toy, audio, lengths, actions, action_lengths, num_rollouts=2)
    assert logprobs.shape == (4,)
    (-logprobs.mean()).backward()
    assert toy.proj.weight.grad is not None

    from lcasr.components.positional_encodings import RotaryPositionalEmbedding
    from lcasr.models.streaming_decoder_asr import CausalDecoderLayer

    torch.manual_seed(7)
    cached_layer = CausalDecoderLayer(d_model=8, n_heads=2, expansion_factor=2)
    cached_layer.eval()

    class RotaryOwner:
        def __init__(self):
            self.rotary_pos_emb = RotaryPositionalEmbedding(dim=4, base=128)

    rotary_owner = RotaryOwner()
    features = torch.randn(2, 5, 8)
    cache = None
    cached_outputs = []
    for step in range(features.size(1)):
        rotary_step = _rotary_emb_fn_for_step(rotary_owner, step, features.device)
        cached_step, cache = _cached_decoder_layer_step(
            layer=cached_layer,
            x_step=features[:, step : step + 1],
            cached_kv=cache,
            rotary_emb_fn=rotary_step,
        )
        cached_outputs.append(cached_step)
        cos, sin = rotary_owner.rotary_pos_emb(step + 1, features.device)
        full_rotary = apply_rotary(cos=cos, sin=sin, learned=rotary_owner.rotary_pos_emb.learned_freq)
        full_prefix = cached_layer(features[:, : step + 1], rotary_emb_fn=full_rotary)
        assert torch.allclose(cached_step, full_prefix[:, -1:], atol=2e-5, rtol=2e-4)
    assert cache is not None and cache.shape[:3] == (2, 5, 2)
    assert len(cached_outputs) == features.size(1)

    class EquivalenceStreaming(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.silence_id = 6
            self.subsampling_factor = 1
            self.input_proj = torch.nn.Linear(80, 8)
            self.prev_token_embedding = torch.nn.Embedding(7, 8)
            self.layers = torch.nn.ModuleList([CausalDecoderLayer(d_model=8, n_heads=2, expansion_factor=2)])
            self.norm = torch.nn.LayerNorm(8)
            self.silence_head = torch.nn.Linear(8, 2)
            self.text_head = torch.nn.Linear(8, 6)
            self.rotary_pos_emb = RotaryPositionalEmbedding(dim=4, base=128)

        def get_silence_id(self):
            return self.silence_id

        def subsampling(self, x, lengths):
            return self.input_proj(x), lengths

        def _rotary_emb_fn(self, length, device):
            cos, sin = self.rotary_pos_emb(length, device)
            return apply_rotary(cos=cos, sin=sin, learned=self.rotary_pos_emb.learned_freq)

        def _combined_logits(self, silence_logits, text_logits):
            return torch.cat([text_logits, silence_logits[:, :1]], dim=-1)

    torch.manual_seed(13)
    equiv_model = EquivalenceStreaming().eval()
    equiv_audio = torch.randn(2, 80, 6)
    equiv_lengths = torch.tensor([6, 4], dtype=torch.long)
    assert rollout_kv_cache_enabled(equiv_model)
    torch.manual_seed(99)
    cached_actions, cached_lengths = sample_streaming_rollouts(
        equiv_model,
        equiv_audio,
        equiv_lengths,
        num_rollouts=3,
        temperature=0.7,
        max_output_frames=5,
        use_kv_cache=True,
    )
    torch.manual_seed(99)
    full_actions, full_lengths = sample_streaming_rollouts(
        equiv_model,
        equiv_audio,
        equiv_lengths,
        num_rollouts=3,
        temperature=0.7,
        max_output_frames=5,
        use_kv_cache=False,
    )
    assert torch.equal(cached_lengths, full_lengths)
    assert torch.equal(cached_actions, full_actions)
    print("Self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", type=str, required=False)
    parser.add_argument("-rm_sched", "--remove_scheduler", action="store_true")
    parser.add_argument("-reset_step", "--reset_step", action="store_true")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=0)
    parser.add_argument("-pin_memory", "--pin_memory", action="store_true")
    parser.add_argument("-prefetch", "--prefetch_factor", type=int, default=1)
    parser.add_argument("-checkpoint_dir", "--checkpoint_dir", type=str, default=None)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=None)
    parser.add_argument("-data_path", "--data_path", type=str, default=None)
    parser.add_argument("-max_records", "--max_records", type=int, default=None)
    parser.add_argument("-max_steps", "--max_steps", type=int, default=None)
    parser.add_argument("-disable_wandb", "--disable_wandb", action="store_true")
    parser.add_argument("--validate_config_only", action="store_true")
    parser.add_argument("--validate_load_model", action="store_true")
    parser.add_argument("--smoke_rollout", action="store_true")
    parser.add_argument("--smoke_cpu", action="store_true")
    parser.add_argument("--smoke_num_rollouts", type=int, default=2)
    parser.add_argument("--smoke_max_output_frames", type=int, default=None)
    parser.add_argument("--smoke_max_records", type=int, default=2)
    parser.add_argument("--self_test", action="store_true")
    parsed = parser.parse_args()

    if parsed.self_test:
        self_test()
    else:
        if parsed.config is None:
            raise ValueError("--config is required unless --self_test is set")
        if parsed.validate_config_only:
            validate_config(parsed)
        elif parsed.smoke_rollout:
            smoke_rollout(parsed)
        else:
            train(parsed)
