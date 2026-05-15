#!/usr/bin/env python3
"""Bounded OGG/torchaudio audit for ROB-89.

This script measures the current Spotify OGG waveform-loading contract needed
before adding the repo-local SpeechCodec-base skeleton. It reads the existing
Spotify spectrogram manifest, maps `.spec.pt` paths back to sibling `.ogg`
sources, samples metadata, and times bounded torchaudio crop loads.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from typing import Any

import torch
import torchaudio


DEFAULT_MANIFEST = "/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json"
DEFAULT_OUTPUT = (
    "/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/"
    "ROB-89/rob89_ogg_torchaudio_audit.json"
)


def load_manifest(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ogg_from_spec(path: str) -> str:
    if not path.endswith(".spec.pt"):
        raise ValueError(f"expected .spec.pt audio path, got {path}")
    return path[: -len(".spec.pt")] + ".ogg"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize_durations(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def find_existing_ogg_records(
    manifest: dict[str, dict[str, Any]],
    min_duration_s: float,
    needed_records: int,
    scan_limit: int,
) -> list[dict[str, Any]]:
    records = []
    scanned = 0
    for key, item in manifest.items():
        scanned += 1
        if scanned > scan_limit:
            break
        audio_path = item.get("audio")
        duration = float(item.get("duration", 0.0) or 0.0)
        if not audio_path or duration < min_duration_s:
            continue
        ogg_path = ogg_from_spec(audio_path)
        if not os.path.exists(ogg_path):
            continue
        records.append(
            {
                "key": key,
                "spec_path": audio_path,
                "ogg_path": ogg_path,
                "txt_path": item.get("txt"),
                "manifest_duration_s": duration,
            }
        )
        if len(records) >= needed_records:
            break
    return records


def read_info(path: str) -> tuple[torchaudio.AudioMetaData, float]:
    start = time.perf_counter()
    info = torchaudio.info(path)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return info, elapsed_ms


def to_codec_input(
    waveform: torch.Tensor, sample_rate: int, target_sample_rate: int
) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_sample_rate
        )
    return waveform.unsqueeze(0).contiguous()


def time_crop_load(
    path: str, seconds: float, source_sample_rate: int, target_sample_rate: int
) -> dict[str, Any]:
    requested_frames = int(round(seconds * source_sample_rate))
    start = time.perf_counter()
    waveform, sample_rate = torchaudio.load(
        path, frame_offset=0, num_frames=requested_frames
    )
    load_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    codec_input = to_codec_input(waveform, sample_rate, target_sample_rate)
    resample_ms = (time.perf_counter() - start) * 1000.0

    return {
        "crop_s": seconds,
        "requested_source_frames": requested_frames,
        "source_sample_rate": sample_rate,
        "loaded_shape": list(waveform.shape),
        "loaded_seconds": waveform.shape[-1] / float(sample_rate),
        "load_ms": load_ms,
        "codec_input_shape": list(codec_input.shape),
        "codec_input_dtype": str(codec_input.dtype).replace("torch.", ""),
        "codec_input_seconds": codec_input.shape[-1] / float(target_sample_rate),
        "resample_to_24khz_ms": resample_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-records", type=int, default=16)
    parser.add_argument("--candidate-scan-limit", type=int, default=512)
    parser.add_argument("--crop-seconds", type=float, nargs="+", default=[5, 30, 120])
    parser.add_argument("--target-sample-rate", type=int, default=24_000)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    existing = find_existing_ogg_records(
        manifest,
        max(args.crop_seconds),
        needed_records=args.metadata_records,
        scan_limit=args.candidate_scan_limit,
    )
    if not existing:
        raise RuntimeError("no manifest records with existing OGG sources found")

    metadata_records = []
    sample_rates: Counter[int] = Counter()
    num_channels: Counter[int] = Counter()
    encodings: Counter[str] = Counter()
    info_elapsed_ms = []
    for record in existing[: args.metadata_records]:
        info, elapsed_ms = read_info(record["ogg_path"])
        duration_s = (
            info.num_frames / float(info.sample_rate) if info.sample_rate else None
        )
        metadata_records.append(
            {
                "key": record["key"],
                "ogg_path": record["ogg_path"],
                "sample_rate": info.sample_rate,
                "num_frames": info.num_frames,
                "num_channels": info.num_channels,
                "bits_per_sample": info.bits_per_sample,
                "encoding": info.encoding,
                "duration_s": duration_s,
                "manifest_duration_s": record["manifest_duration_s"],
                "info_ms": elapsed_ms,
            }
        )
        sample_rates[info.sample_rate] += 1
        num_channels[info.num_channels] += 1
        encodings[info.encoding] += 1
        info_elapsed_ms.append(elapsed_ms)

    selected = existing[0]
    selected_info, selected_info_ms = read_info(selected["ogg_path"])
    crop_results = [
        time_crop_load(
            selected["ogg_path"],
            seconds,
            selected_info.sample_rate,
            args.target_sample_rate,
        )
        for seconds in args.crop_seconds
    ]

    result = {
        "manifest_path": args.manifest,
        "manifest_records": len(manifest),
        "candidate_scan_limit": args.candidate_scan_limit,
        "existing_ogg_records_found_in_bounded_scan": len(existing),
        "source_path_rule": "manifest audio .spec.pt path with .spec.pt replaced by .ogg",
        "metadata_sample_size": len(metadata_records),
        "sample_rate_counts": dict(sorted(sample_rates.items())),
        "channel_counts": dict(sorted(num_channels.items())),
        "encoding_counts": dict(sorted(encodings.items())),
        "metadata_info_ms": summarize_durations(info_elapsed_ms),
        "duration_metadata_read_without_decode": all(
            item["num_frames"] > 0 and item["sample_rate"] > 0
            for item in metadata_records
        ),
        "selected_record": {
            **selected,
            "torchaudio_info_sample_rate": selected_info.sample_rate,
            "torchaudio_info_num_frames": selected_info.num_frames,
            "torchaudio_info_duration_s": (
                selected_info.num_frames / float(selected_info.sample_rate)
                if selected_info.sample_rate
                else None
            ),
            "torchaudio_info_ms": selected_info_ms,
        },
        "crop_results": crop_results,
        "proposed_resampling_policy": (
            "load OGG with torchaudio, collapse to mono by channel mean, resample "
            "to 24000 Hz before SpeechCodec-base, and crop by source-sample "
            "frame counts before resampling"
        ),
    }

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
