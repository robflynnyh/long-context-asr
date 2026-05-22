#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def lr_token(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("+", "")


def load_checkpoint_config(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise SystemExit(f"{path} has no config")
    config = OmegaConf.to_container(checkpoint["config"], resolve=True)
    if "model" not in config:
        raise SystemExit(f"{path} config has no model section")
    return config


def base_config(args, source_config, learning_rate, run_label):
    batch_size = args.smoke_batch_size if args.smoke else args.batch_size
    max_epochs = 1 if args.smoke else args.max_epochs
    run_id = Path(args.run_dir).name
    diagnostics_path = Path(args.run_dir) / "diagnostics" / f"{run_label}.jsonl"
    model_config = dict(source_config["model"])

    return {
        "model_class": source_config.get("model_class", "EncDecSconformerV2"),
        "description": (
            "ROB-125 frozen supervised encoder CTC sanity probe using the ROB-119-style "
            f"hidden-state weighted sum and {args.bilstm_num_layers}-layer "
            f"BiLSTM({args.bilstm_hidden_size}) head."
        ),
        "model": model_config,
        "probe": {
            "ssl_checkpoint": args.supervised_checkpoint,
            "source_checkpoint": args.supervised_checkpoint,
            "head": "bilstm",
            "hidden_state_weighted_sum": True,
            "num_hidden_states": args.num_hidden_states,
            "load_decoder_from_ssl": True,
            "trainable_prefixes": ["decoder.", "weighted_sum."],
            "bilstm_hidden_size": args.bilstm_hidden_size,
            "bilstm_num_layers": args.bilstm_num_layers,
            "bilstm_dropout": args.bilstm_dropout,
        },
        "optimizer": {"name": "madgrad", "args": {"lr": learning_rate}},
        "scheduler": {"name": args.scheduler, "warmup_steps": args.warmup_steps},
        "audio_chunking": {"size": args.seq_len, "overlap": 0},
        "wandb": {
            "use": not args.disable_wandb and (not args.smoke or args.enable_smoke_wandb),
            "project_name": "rob125_supervised_ctc_probe",
            "name": f"{run_id}_{run_label}_supervised_weighted_bilstm_ctc_probe",
            "id": "",
            "dir": str(Path(args.run_dir) / "wandb"),
            "watch_model": False,
            "update_config_with_wandb_id": False,
        },
        "checkpointing": {
            "dir": str(Path(args.checkpoint_root) / run_label),
            "save_every_n_steps": args.save_every_n_steps,
        },
        "data": {"path": args.train_manifest},
        "training": {
            "batch_size": batch_size,
            "backprop_every": 1,
            "backwards_every": 1,
            "max_epochs": max_epochs,
            "clip_value": 0.8,
            "random_seed": args.random_seed,
            "dtype": args.dtype,
            "ctc_zero_infinity": True,
            "diagnostics_path": str(diagnostics_path),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument(
        "--supervised-checkpoint",
        default="/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt",
    )
    parser.add_argument("--known-good-evidence", default="ROB-81 TEDLIUM eval WER 0.08258018784334574 over 28215 words")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable-smoke-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rates")
    parser.add_argument("--scheduler", choices=["cosine", "constant"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-n-steps", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--num-hidden-states", type=int, default=3)
    parser.add_argument("--bilstm-hidden-size", type=int, default=1024)
    parser.add_argument("--bilstm-num-layers", type=int, default=2)
    parser.add_argument("--bilstm-dropout", type=float, default=0.2)
    args = parser.parse_args()

    supervised_checkpoint = Path(args.supervised_checkpoint)
    if not supervised_checkpoint.exists():
        raise SystemExit(f"missing supervised checkpoint: {supervised_checkpoint}")
    source_config = load_checkpoint_config(supervised_checkpoint)
    learning_rates = parse_csv(args.learning_rates, float) or [args.learning_rate]

    config_dir = Path(args.run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)

    runs = []
    single_lr = len(learning_rates) == 1
    for learning_rate in learning_rates:
        run_label = "floras81" if single_lr else f"floras81_lr{lr_token(learning_rate)}"
        config = OmegaConf.create(base_config(args, source_config, learning_rate, run_label))
        config_path = config_dir / f"rob125_{run_label}_supervised_weighted_bilstm_ctc_probe.yaml"
        OmegaConf.save(config=config, f=config_path)
        runs.append(
            {
                "label": run_label,
                "base_label": "floras81",
                "learning_rate": learning_rate,
                "scheduler": args.scheduler,
                "warmup_steps": args.warmup_steps,
                "probe_head": "bilstm",
                "hidden_state_weighted_sum": True,
                "num_hidden_states": args.num_hidden_states,
                "max_epochs": config.training.max_epochs,
                "source_checkpoint": args.supervised_checkpoint,
                "local_checkpoint": args.supervised_checkpoint,
                "known_good_evidence": args.known_good_evidence,
                "train_config": str(config_path),
                "checkpoint_dir": str(Path(args.checkpoint_root) / run_label),
                "diagnostics_path": config.training.diagnostics_path,
            }
        )

    manifest = {
        "issue": "ROB-125",
        "mode": "smoke" if args.smoke else "full",
        "run_dir": args.run_dir,
        "train_manifest": args.train_manifest,
        "checkpoint_root": args.checkpoint_root,
        "max_epochs": 1 if args.smoke else args.max_epochs,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "comparison_note": "TEDLIUM transfer sanity probe with known-good supervised encoder features.",
        "runs": runs,
    }
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.manifest_out)


if __name__ == "__main__":
    main()
