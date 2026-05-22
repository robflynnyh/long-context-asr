#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


DEFAULT_SOURCE_MANIFEST = (
    "/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-125/"
    "rob125-full-supervised-4epoch-20260522T165541Z/tedlium_train_manifest.json"
)
DEFAULT_CHECKPOINT = (
    "/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/"
    "supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt"
)


def load_checkpoint_config(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if "config" not in checkpoint:
        raise SystemExit(f"{path} has no config")
    return OmegaConf.to_container(checkpoint["config"], resolve=True)


def select_record(manifest: dict, requested: str | None):
    if requested:
        if requested not in manifest:
            raise SystemExit(f"record {requested} not found in source manifest")
        return requested, manifest[requested]
    return min(manifest.items(), key=lambda item: float(item[1]["duration"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--run-manifest-out", required=True)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--record-id")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--head", choices=("linear", "bilstm"), default="bilstm")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--save-every-n-steps", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=1234)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
    record_id, record = select_record(source_manifest, args.record_id)
    one_record_manifest = run_dir / "one_record_train_manifest.json"
    one_record_manifest.write_text(json.dumps({record_id: record}, indent=2), encoding="utf-8")

    source_config = load_checkpoint_config(args.checkpoint)
    model_config = dict(source_config["model"])
    config = {
        "model_class": source_config.get("model_class", "EncDecSconformerV2"),
        "description": "ROB-128 one-record supervised-feature CTC probe overfit check.",
        "model": model_config,
        "probe": {
            "ssl_checkpoint": args.checkpoint,
            "source_checkpoint": args.checkpoint,
            "head": args.head,
            "load_decoder_from_ssl": True,
            "trainable_prefixes": ["decoder."],
            "bilstm_hidden_size": 1024,
            "bilstm_num_layers": 2,
            "bilstm_dropout": 0.2,
        },
        "optimizer": {"name": "madgrad", "args": {"lr": args.learning_rate}},
        "scheduler": {"name": "constant", "warmup_steps": 0},
        "audio_chunking": {"size": args.seq_len, "overlap": 0},
        "wandb": {
            "use": False,
            "project_name": "rob128_probe_overfit",
            "name": f"{run_dir.name}_{record_id}_{args.head}",
            "id": "",
            "dir": str(run_dir / "wandb"),
            "watch_model": False,
            "update_config_with_wandb_id": False,
        },
        "checkpointing": {
            "dir": str(checkpoint_dir),
            "save_every_n_steps": args.save_every_n_steps,
        },
        "data": {"path": str(one_record_manifest)},
        "training": {
            "batch_size": args.batch_size,
            "backprop_every": 1,
            "backwards_every": 1,
            "max_epochs": args.max_epochs,
            "clip_value": 0.8,
            "random_seed": args.random_seed,
            "dtype": "float32",
            "ctc_zero_infinity": True,
            "diagnostics_path": str(run_dir / "diagnostics.jsonl"),
        },
    }

    OmegaConf.save(config=OmegaConf.create(config), f=args.config_out)
    run_manifest = {
        "issue": "ROB-128",
        "run_dir": str(run_dir),
        "record_id": record_id,
        "record": record,
        "source_manifest": args.source_manifest,
        "one_record_manifest": str(one_record_manifest),
        "source_checkpoint": args.checkpoint,
        "train_config": args.config_out,
        "checkpoint_dir": str(checkpoint_dir),
        "head": args.head,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "seq_len": args.seq_len,
        "diagnostics_path": str(run_dir / "diagnostics.jsonl"),
    }
    Path(args.run_manifest_out).write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
