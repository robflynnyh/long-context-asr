#!/usr/bin/env python3
"""Prepare ROB-192 streaming-decoder Floras finetuning config and manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import sentencepiece as spm
from omegaconf import OmegaConf

from symphony.rob81_prepare_floras_manifest import (
    normalize_timestamps,
    oov_words,
    safe_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="exp/configs/streaming_decoder_asr_100m.yaml")
    parser.add_argument("--mapping", default="/users/acp21rjf/align_floras50/tmp/mapping.json")
    parser.add_argument("--tokenizer", default="lcasr/artifacts/tokenizer.model")
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--text-output-dir", required=True)
    parser.add_argument("--manifest-summary-json", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--wandb-project-name", default="floras50_streaming_decoder_supervised")
    parser.add_argument("--wandb-name", default="rob192_rope_streaming_decoder_floras50_lr5e-5_12ep")
    parser.add_argument("--batch-size", type=int, default=88)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-every-n-steps", type=int, default=50000)
    parser.add_argument("--subsampling-factor", type=int, default=8)
    parser.add_argument("--rotary-base-freq", type=int, default=1_500_000)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--debug-generate-every-records", type=int, default=500)
    parser.add_argument("--debug-generate-max-frames", type=int, default=0)
    parser.add_argument("--validate-path-limit", type=int, default=50)
    parser.add_argument("--validate-all-paths", action="store_true")
    parser.add_argument("--no-validate-paths", action="store_true")
    parser.add_argument("--manifest-limit", type=int, default=0)
    parser.add_argument("--unk-id", type=int, default=1)
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def validate_paths(pairs: Dict[str, Dict[str, Any]], limit: int | None) -> int:
    missing = []
    checked = 0
    for key, record in pairs.items():
        if limit is not None and checked >= limit:
            break
        checked += 1
        for kind in ("audio", "txt"):
            path = record.get(kind)
            if not path or not os.path.exists(path):
                missing.append((key, kind, path or "<missing>"))
        if len(missing) >= 10:
            break
    if missing:
        detail = "\n".join(f"{key} {kind}: {path}" for key, kind, path in missing)
        raise FileNotFoundError(f"missing Floras files:\n{detail}")
    return checked


def prepare_manifest(args: argparse.Namespace) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    mapping = load_json(args.mapping)
    text_output_dir = Path(args.text_output_dir)
    text_output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_summary_json).parent.mkdir(parents=True, exist_ok=True)

    items = list(mapping.items())
    if args.manifest_limit > 0:
        items = items[: args.manifest_limit]

    filtered_mapping: Dict[str, Dict[str, Any]] = {}
    unk_cache: Dict[str, bool] = {}
    stats: Dict[str, Any] = {
        "input_records": 0,
        "kept_records": 0,
        "dropped_missing_text": 0,
        "dropped_empty_after_normalization": 0,
        "dropped_oov_after_normalization": 0,
        "unique_words_checked": 0,
        "examples": [],
    }

    for record_id, record in items:
        stats["input_records"] += 1
        txt_path = str(record.get("txt", ""))
        if not txt_path or not os.path.exists(txt_path):
            stats["dropped_missing_text"] += 1
            continue

        payload = load_json(txt_path)
        timestamps = payload.get("word_timestamps", [])
        normalized_timestamps, normalized_words = normalize_timestamps(timestamps)
        if not normalized_words:
            stats["dropped_empty_after_normalization"] += 1
            continue

        record_oov_words = oov_words(tokenizer, normalized_words, args.unk_id, unk_cache)
        if record_oov_words:
            stats["dropped_oov_after_normalization"] += 1
            if len(stats["examples"]) < args.examples:
                stats["examples"].append(
                    {"record_id": record_id, "txt": txt_path, "oov_words": record_oov_words[:20]}
                )
            continue

        normalized_payload = dict(payload)
        normalized_payload["word_timestamps"] = normalized_timestamps
        normalized_txt_path = text_output_dir / f"{safe_name(record_id)}.json"
        write_json(str(normalized_txt_path), normalized_payload)

        filtered_record = dict(record)
        filtered_record["txt"] = str(normalized_txt_path)
        filtered_mapping[record_id] = filtered_record
        stats["kept_records"] += 1

    stats["unique_words_checked"] = len(unk_cache)
    write_json(args.manifest_out, filtered_mapping)
    write_json(args.manifest_summary_json, stats)
    return filtered_mapping, stats


def write_runtime_config(args: argparse.Namespace, records: Dict[str, Dict[str, Any]]) -> None:
    if not os.path.exists(args.pretrained_checkpoint):
        raise FileNotFoundError(f"pretrained checkpoint does not exist: {args.pretrained_checkpoint}")

    config = OmegaConf.load(args.base_config)
    config.model.use_rotary = True
    config.model.rotary_base_freq = args.rotary_base_freq
    config.model.rotary_interpolation_factor = config.model.get("rotary_interpolation_factor", 1.0)
    config.model.subsampling_factor = args.subsampling_factor
    config.data.path = args.manifest_out
    if "max_records" in config.data:
        del config.data.max_records
    config.checkpointing.dir = args.checkpoint_dir
    config.checkpointing.pretrained = args.pretrained_checkpoint
    config.checkpointing.save_every_n_steps = args.save_every_n_steps
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.optimizer.args.lr = args.learning_rate
    config.streaming.delay_seconds = args.delay_seconds
    config.training.debug_generation = {
        "enabled": args.debug_generate_every_records > 0,
        "every_records": args.debug_generate_every_records,
        "max_frames": args.debug_generate_max_frames,
    }
    if "max_steps" in config.training:
        del config.training.max_steps
    config.wandb.use = True
    config.wandb.project_name = args.wandb_project_name
    config.wandb.name = args.wandb_name
    config.wandb.id = ""
    config.wandb.dir = args.wandb_dir
    config.wandb.update_config_with_wandb_id = False

    os.makedirs(os.path.dirname(args.config_out), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.wandb_dir, exist_ok=True)
    OmegaConf.save(config=config, f=args.config_out)

    durations = [float(record.get("duration", 0.0)) for record in records.values()]
    lines = [
        f"records={len(records)}",
        f"total_duration_hours={sum(durations) / 3600.0:.2f}",
        f"mapping={args.mapping}",
        f"manifest={args.manifest_out}",
        f"manifest_summary_json={args.manifest_summary_json}",
        f"runtime_config={args.config_out}",
        f"checkpoint_dir={args.checkpoint_dir}",
        f"pretrained_checkpoint={args.pretrained_checkpoint}",
        f"wandb_dir={args.wandb_dir}",
        f"wandb_project_name={args.wandb_project_name}",
        f"wandb_name={args.wandb_name}",
        f"batch_size={args.batch_size}",
        f"max_epochs={args.max_epochs}",
        f"learning_rate={config.optimizer.args.lr}",
        f"save_every_n_steps={config.checkpointing.save_every_n_steps}",
        f"subsampling_factor={config.model.subsampling_factor}",
        f"subsampling_conv_chunking_factor={config.model.get('subsampling_conv_chunking_factor', '')}",
        f"use_rotary={config.model.use_rotary}",
        f"rotary_base_freq={config.model.rotary_base_freq}",
        f"rotary_interpolation_factor={config.model.rotary_interpolation_factor}",
        f"delay_seconds={config.streaming.delay_seconds}",
        f"buffer_seconds={config.streaming.buffer_seconds}",
        f"debug_generate_every_records={args.debug_generate_every_records}",
        f"debug_generate_max_frames={args.debug_generate_max_frames}",
    ]
    with open(args.summary_out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    args = parse_args()
    records, manifest_stats = prepare_manifest(args)
    if not records:
        raise RuntimeError(f"filtered manifest is empty: {args.manifest_out}")

    validate_limit = None if args.validate_all_paths else args.validate_path_limit
    checked_paths = 0
    if not args.no_validate_paths:
        checked_paths = validate_paths(records, validate_limit)

    write_runtime_config(args, records)
    print(json.dumps(manifest_stats, indent=2, ensure_ascii=False))
    print(f"validated_path_records={checked_paths}")


if __name__ == "__main__":
    main()
