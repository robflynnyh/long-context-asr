#!/usr/bin/env python3
"""CPU-only ROB-207 BEST-RQ codebook-utilization sanity check."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torchaudio
from vector_quantize_pytorch import RandomProjectionQuantizer

from lcasr.utils.audio_tools import HOP_LENGTH, SR, WIN_LENGTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare BEST-RQ random-projection codebook utilization for current "
            "power-mel features and a log-mel variant on the same recordings."
        )
    )
    parser.add_argument(
        "--pairs",
        default="/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/"
            "rob100_papermask_p012_l4_sc_off_20260520/step_105360.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/"
            "ROB-207/codebook-util-logmel"
        ),
    )
    parser.add_argument("--max-records", type=int, default=24)
    parser.add_argument("--min-duration", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=180.0)
    parser.add_argument("--max-feature-frames", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=207)
    parser.add_argument("--codebook-size", type=int, default=8192)
    parser.add_argument("--codebook-dim", type=int, default=16)
    parser.add_argument("--feat-in", type=int, default=80)
    parser.add_argument("--downsampling-factor", type=int, default=8)
    parser.add_argument("--log-eps", type=float, default=1e-10)
    parser.add_argument("--quantize-batch-frames", type=int, default=8192)
    return parser.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_per_recording(spec: torch.Tensor) -> torch.Tensor:
    return (spec - spec.mean(-1, keepdim=True)) / spec.std(-1, keepdim=True)


def resolve_ogg_path(spec_path: str) -> str:
    if spec_path.endswith(".spec.pt"):
        return spec_path[: -len(".spec.pt")] + ".ogg"
    return os.path.splitext(spec_path)[0] + ".ogg"


def make_mel_transform() -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_fft=2 ** math.ceil(math.log2(WIN_LENGTH)),
        n_mels=80,
        normalized=False,
    )


def load_waveform(path: str) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.ndim == 1:
        waveform = waveform[None]
    else:
        waveform = waveform[:1]
    if sample_rate != SR:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SR,
        )(waveform)
        sample_rate = SR
    return waveform, sample_rate


def compute_feature_variants(
    ogg_path: str,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    log_eps: float,
    max_feature_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    waveform, _ = load_waveform(ogg_path)
    mel = mel_transform(waveform).float()
    if max_feature_frames > 0:
        mel = mel[..., :max_feature_frames]
    current_power = normalize_per_recording(mel)
    log_mel = normalize_per_recording(torch.log(mel.clamp_min(log_eps)))
    return current_power, log_mel


def load_saved_feature(spec_path: str, max_feature_frames: int) -> torch.Tensor:
    spec = torch_load_cpu(spec_path).float()
    if spec.ndim == 2:
        spec = spec[None]
    if max_feature_frames > 0:
        spec = spec[..., :max_feature_frames]
    return spec


def stack_frames(spec: torch.Tensor, downsampling_factor: int) -> torch.Tensor:
    if spec.ndim != 3 or spec.shape[0] != 1:
        raise ValueError(f"expected feature shape [1, 80, T], got {tuple(spec.shape)}")
    usable_frames = (spec.shape[-1] // downsampling_factor) * downsampling_factor
    spec = spec[:, :, :usable_frames]
    if usable_frames == 0:
        return torch.empty(0, spec.shape[1] * downsampling_factor)
    time_major = spec.squeeze(0).transpose(0, 1).contiguous()
    return time_major.view(-1, spec.shape[1] * downsampling_factor)


def load_quantizer(args: argparse.Namespace) -> RandomProjectionQuantizer:
    quantizer = RandomProjectionQuantizer(
        dim=args.feat_in * args.downsampling_factor,
        num_codebooks=1,
        codebook_dim=args.codebook_dim,
        codebook_size=args.codebook_size,
    )
    checkpoint = torch_load_cpu(args.checkpoint)
    state_dict = checkpoint["model"]
    quantizer_state = {
        key.removeprefix("quantizer."): value
        for key, value in state_dict.items()
        if key.startswith("quantizer.")
    }
    quantizer.load_state_dict(quantizer_state, strict=True)
    quantizer.eval()
    return quantizer


def quantize_frames(
    quantizer: RandomProjectionQuantizer,
    frames: torch.Tensor,
    batch_frames: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_frames):
            chunk = frames[start : start + batch_frames].float()
            outputs.append(quantizer(chunk[None]).reshape(-1).cpu())
    if not outputs:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(outputs).long()


def update_counts(counts: torch.Tensor, indices: torch.Tensor) -> None:
    if indices.numel() == 0:
        return
    counts += torch.bincount(indices, minlength=counts.numel()).cpu()


def summarize_counts(counts: torch.Tensor) -> Dict[str, float]:
    total = int(counts.sum().item())
    nonzero = counts[counts > 0].float()
    if total == 0:
        return {
            "frames": 0,
            "unique_codes": 0,
            "usage_fraction": 0.0,
            "entropy_nats": 0.0,
            "entropy_norm": 0.0,
            "perplexity": 0.0,
            "perplexity_fraction": 0.0,
            "top1_mass": 0.0,
            "top10_mass": 0.0,
            "top100_mass": 0.0,
        }
    probs = nonzero / total
    entropy = float(-(probs * probs.log()).sum().item())
    sorted_counts = torch.sort(counts, descending=True).values.float()
    codebook_size = counts.numel()
    return {
        "frames": total,
        "unique_codes": int((counts > 0).sum().item()),
        "usage_fraction": float((counts > 0).float().mean().item()),
        "entropy_nats": entropy,
        "entropy_norm": entropy / math.log(codebook_size),
        "perplexity": math.exp(entropy),
        "perplexity_fraction": math.exp(entropy) / codebook_size,
        "top1_mass": float(sorted_counts[:1].sum().item() / total),
        "top10_mass": float(sorted_counts[:10].sum().item() / total),
        "top100_mass": float(sorted_counts[:100].sum().item() / total),
    }


def js_divergence_bits(counts_a: torch.Tensor, counts_b: torch.Tensor) -> float:
    total_a = counts_a.sum().item()
    total_b = counts_b.sum().item()
    if total_a == 0 or total_b == 0:
        return 0.0
    p = counts_a.float() / total_a
    q = counts_b.float() / total_b
    m = 0.5 * (p + q)

    def kl_bits(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        return (x[mask] * (x[mask] / y[mask]).log2()).sum()

    return float(0.5 * kl_bits(p, m).item() + 0.5 * kl_bits(q, m).item())


def top_code_overlap(counts_a: torch.Tensor, counts_b: torch.Tensor, k: int) -> int:
    top_a = set(torch.topk(counts_a, k=min(k, counts_a.numel())).indices.tolist())
    top_b = set(torch.topk(counts_b, k=min(k, counts_b.numel())).indices.tolist())
    return len(top_a & top_b)


def select_records(args: argparse.Namespace) -> List[Tuple[str, Dict]]:
    pairs = load_json(args.pairs)
    candidates = []
    for record_id, entry in pairs.items():
        duration = float(entry.get("duration", -1.0))
        if duration < args.min_duration or duration > args.max_duration:
            continue
        spec_path = entry.get("audio")
        if not spec_path:
            continue
        candidates.append((record_id, entry))
    random.Random(args.seed).shuffle(candidates)
    return candidates


def write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def format_metric_row(name: str, summary: Dict[str, float]) -> str:
    return (
        f"| {name} | {summary['frames']} | {summary['unique_codes']} | "
        f"{summary['usage_fraction']:.4f} | {summary['entropy_norm']:.4f} | "
        f"{summary['perplexity']:.1f} | {summary['top1_mass']:.4f} | "
        f"{summary['top10_mass']:.4f} | {summary['top100_mass']:.4f} |"
    )


def write_markdown(path: Path, payload: Dict) -> None:
    summaries = payload["summaries"]
    comparisons = payload["comparisons"]
    lines = [
        "# ROB-207 Codebook Utilization Sanity Check",
        "",
        f"Records processed: {payload['records_processed']}",
        f"Total aligned stacked frames: {payload['aligned_stacked_frames']}",
        f"Seed: {payload['args']['seed']}",
        f"Duration window: {payload['args']['min_duration']}s to {payload['args']['max_duration']}s",
        f"Max feature frames per recording: {payload['args']['max_feature_frames']}",
        "",
        "| Variant | Frames | Unique codes | Usage frac | Entropy norm | Perplexity | Top1 mass | Top10 mass | Top100 mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        format_metric_row("saved current .spec.pt", summaries["saved_current"]),
        format_metric_row("recomputed power-mel norm", summaries["power_mel_norm"]),
        format_metric_row("log-mel norm", summaries["log_mel_norm"]),
        "",
        "## Comparisons",
        "",
        f"- power/log label agreement: {comparisons['power_log_label_agreement']:.6f}",
        f"- saved/power label agreement: {comparisons['saved_power_label_agreement']:.6f}",
        f"- power/log JS divergence bits: {comparisons['power_log_jsd_bits']:.6f}",
        f"- saved/power JS divergence bits: {comparisons['saved_power_jsd_bits']:.6f}",
        f"- power/log top-100 overlap: {comparisons['power_log_top100_overlap']}",
        f"- saved/power top-100 overlap: {comparisons['saved_power_top100_overlap']}",
        f"- mean abs saved-vs-power feature diff: {comparisons['mean_abs_saved_power_feature_diff']:.6f}",
        "",
        "## Files",
        "",
        f"- summary JSON: {payload['files']['summary_json']}",
        f"- per-recording CSV: {payload['files']['per_recording_csv']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quantizer = load_quantizer(args)
    mel_transform = make_mel_transform()

    counts = {
        "saved_current": torch.zeros(args.codebook_size, dtype=torch.long),
        "power_mel_norm": torch.zeros(args.codebook_size, dtype=torch.long),
        "log_mel_norm": torch.zeros(args.codebook_size, dtype=torch.long),
    }
    total_agreement = {"power_log": 0, "saved_power": 0}
    total_aligned = 0
    total_saved_power_feature_abs = 0.0
    total_saved_power_feature_elements = 0
    per_recording_rows = []

    for record_id, entry in select_records(args):
        if len(per_recording_rows) >= args.max_records:
            break
        spec_path = entry["audio"]
        ogg_path = resolve_ogg_path(spec_path)
        if not os.path.exists(spec_path) or not os.path.exists(ogg_path):
            continue

        try:
            saved = load_saved_feature(spec_path, args.max_feature_frames)
            power, log_mel = compute_feature_variants(
                ogg_path=ogg_path,
                mel_transform=mel_transform,
                log_eps=args.log_eps,
                max_feature_frames=args.max_feature_frames,
            )
        except Exception as exc:
            per_recording_rows.append(
                {
                    "id": record_id,
                    "duration": entry.get("duration", ""),
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
            )
            continue

        min_t = min(saved.shape[-1], power.shape[-1], log_mel.shape[-1])
        saved, power, log_mel = saved[..., :min_t], power[..., :min_t], log_mel[..., :min_t]
        saved_frames = stack_frames(saved, args.downsampling_factor)
        power_frames = stack_frames(power, args.downsampling_factor)
        log_frames = stack_frames(log_mel, args.downsampling_factor)
        frame_count = min(saved_frames.shape[0], power_frames.shape[0], log_frames.shape[0])
        if frame_count == 0:
            continue
        saved_frames = saved_frames[:frame_count]
        power_frames = power_frames[:frame_count]
        log_frames = log_frames[:frame_count]

        saved_idx = quantize_frames(quantizer, saved_frames, args.quantize_batch_frames)
        power_idx = quantize_frames(quantizer, power_frames, args.quantize_batch_frames)
        log_idx = quantize_frames(quantizer, log_frames, args.quantize_batch_frames)

        update_counts(counts["saved_current"], saved_idx)
        update_counts(counts["power_mel_norm"], power_idx)
        update_counts(counts["log_mel_norm"], log_idx)

        power_log_agree = int((power_idx == log_idx).sum().item())
        saved_power_agree = int((saved_idx == power_idx).sum().item())
        total_agreement["power_log"] += power_log_agree
        total_agreement["saved_power"] += saved_power_agree
        total_aligned += frame_count

        feature_elements = min_t * saved.shape[1]
        feature_abs = float((saved - power).abs().sum().item())
        total_saved_power_feature_abs += feature_abs
        total_saved_power_feature_elements += feature_elements

        per_recording_rows.append(
            {
                "id": record_id,
                "duration": entry.get("duration", ""),
                "status": "ok",
                "stacked_frames": frame_count,
                "saved_unique": int(saved_idx.unique().numel()),
                "power_unique": int(power_idx.unique().numel()),
                "log_unique": int(log_idx.unique().numel()),
                "power_log_agreement": power_log_agree / frame_count,
                "saved_power_agreement": saved_power_agree / frame_count,
                "mean_abs_saved_power_feature_diff": feature_abs / feature_elements,
                "spec_path": spec_path,
                "ogg_path": ogg_path,
            }
        )

    per_recording_csv = output_dir / "per_recording.csv"
    fieldnames = sorted({key for row in per_recording_rows for key in row.keys()})
    with per_recording_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_recording_rows)

    summaries = {name: summarize_counts(value) for name, value in counts.items()}
    comparisons = {
        "power_log_label_agreement": (
            total_agreement["power_log"] / total_aligned if total_aligned else 0.0
        ),
        "saved_power_label_agreement": (
            total_agreement["saved_power"] / total_aligned if total_aligned else 0.0
        ),
        "power_log_jsd_bits": js_divergence_bits(
            counts["power_mel_norm"], counts["log_mel_norm"]
        ),
        "saved_power_jsd_bits": js_divergence_bits(
            counts["saved_current"], counts["power_mel_norm"]
        ),
        "power_log_top100_overlap": top_code_overlap(
            counts["power_mel_norm"], counts["log_mel_norm"], 100
        ),
        "saved_power_top100_overlap": top_code_overlap(
            counts["saved_current"], counts["power_mel_norm"], 100
        ),
        "mean_abs_saved_power_feature_diff": (
            total_saved_power_feature_abs / total_saved_power_feature_elements
            if total_saved_power_feature_elements
            else 0.0
        ),
    }

    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    payload = {
        "args": vars(args),
        "records_processed": sum(1 for row in per_recording_rows if row.get("status") == "ok"),
        "aligned_stacked_frames": total_aligned,
        "summaries": summaries,
        "comparisons": comparisons,
        "files": {
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "per_recording_csv": str(per_recording_csv),
        },
    }
    write_json(summary_json, payload)
    write_markdown(summary_md, payload)

    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
