#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


CHECKPOINTS = [
    ("primary", "step_105360.pt"),
    ("backup", "step_99264.pt"),
]
CHECKPOINT_MAP = dict(CHECKPOINTS)


def parse_csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def lr_token(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("+", "")


def base_config(args, label, checkpoint_file, learning_rate, run_label):
    batch_size = args.smoke_batch_size if args.smoke else args.batch_size
    max_epochs = 1 if args.smoke else args.max_epochs
    run_id = Path(args.run_dir).name
    local_checkpoint = Path(args.checkpoint_cache) / checkpoint_file
    diagnostics_path = Path(args.run_dir) / "diagnostics" / f"{run_label}.jsonl"
    return {
        "model_class": "SCConformerXL",
        "description": (
            "ROB-119 frozen ROB-100 BEST-RQ CTC probe with trainable hidden-state "
            f"weighted sum and {args.bilstm_num_layers}-layer BiLSTM({args.bilstm_hidden_size}) head."
        ),
        "model": {
            "feat_in": 80,
            "n_layers": 6,
            "d_model": 768,
            "n_heads": 6,
            "head_dim": 128,
            "dropout_ff": 0.0,
            "dropout_attn": 0.0,
            "dropout_conv": 0.0,
            "subsampling_factor": 8,
            "subsampling": "dw_striding",
            "subsampling_act": "silu",
            "subsampling_conv_channels": 256,
            "self_condition_subsampling": False,
            "subsampling_norm_out": False,
            "conv_kernel_size": 9,
            "qk_rms_norm": False,
            "shift_kvs": False,
            "self_conditioning": False,
            "gated_sc": False,
            "decoder_norm": True,
            "use_rotary": True,
            "encoder_mode": "conformer",
            "default_norm": "layer_norm",
            "sandwich_norm": False,
            "bias_in_ff": False,
            "checkpoint_every_n_layers": 0,
            "rotary_base_freq": 1500000,
            "flash_attn": True,
        },
        "probe": {
            "ssl_checkpoint": str(local_checkpoint),
            "source_checkpoint": f"{args.source_checkpoint_dir}/{checkpoint_file}",
            "head": "bilstm",
            "hidden_state_weighted_sum": True,
            "num_hidden_states": args.num_hidden_states,
            "load_decoder_from_ssl": False,
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
            "project_name": "rob119_bestrq_ctc_probe",
            "name": f"{run_id}_{run_label}_weighted_bilstm_ctc_probe",
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
    parser.add_argument("--checkpoint-cache", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument(
        "--source-checkpoint-dir",
        default="/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable-smoke-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rates")
    parser.add_argument("--checkpoint-labels", default="primary")
    parser.add_argument("--scheduler", choices=["cosine", "constant"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-n-steps", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--num-hidden-states", type=int, default=6)
    parser.add_argument("--bilstm-hidden-size", type=int, default=1024)
    parser.add_argument("--bilstm-num-layers", type=int, default=2)
    parser.add_argument("--bilstm-dropout", type=float, default=0.2)
    args = parser.parse_args()

    selected_labels = parse_csv(args.checkpoint_labels) or ["primary"]
    unknown = [label for label in selected_labels if label not in CHECKPOINT_MAP]
    if unknown:
        raise SystemExit(f"unknown checkpoint label(s): {', '.join(unknown)}")
    labels = [(label, CHECKPOINT_MAP[label]) for label in selected_labels]
    if args.smoke:
        labels = labels[:1]
    learning_rates = parse_csv(args.learning_rates, float) or [args.learning_rate]

    config_dir = Path(args.run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)

    runs = []
    single_lr = len(learning_rates) == 1
    for label, checkpoint_file in labels:
        for learning_rate in learning_rates:
            run_label = label if single_lr else f"{label}_lr{lr_token(learning_rate)}"
            config = OmegaConf.create(base_config(args, label, checkpoint_file, learning_rate, run_label))
            config_path = config_dir / f"rob119_{run_label}_weighted_bilstm_ctc_probe.yaml"
            OmegaConf.save(config=config, f=config_path)
            runs.append(
                {
                    "label": run_label,
                    "base_label": label,
                    "learning_rate": learning_rate,
                    "scheduler": args.scheduler,
                    "warmup_steps": args.warmup_steps,
                    "checkpoint_file": checkpoint_file,
                    "probe_head": "bilstm",
                    "hidden_state_weighted_sum": True,
                    "num_hidden_states": args.num_hidden_states,
                    "max_epochs": config.training.max_epochs,
                    "source_checkpoint": f"{args.source_checkpoint_dir}/{checkpoint_file}",
                    "local_checkpoint": str(Path(args.checkpoint_cache) / checkpoint_file),
                    "train_config": str(config_path),
                    "checkpoint_dir": str(Path(args.checkpoint_root) / run_label),
                    "diagnostics_path": config.training.diagnostics_path,
                }
            )

    manifest = {
        "issue": "ROB-119",
        "mode": "smoke" if args.smoke else "full",
        "run_dir": args.run_dir,
        "train_manifest": args.train_manifest,
        "checkpoint_cache": args.checkpoint_cache,
        "checkpoint_root": args.checkpoint_root,
        "max_epochs": 1 if args.smoke else args.max_epochs,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "comparison_note": "TEDLIUM transfer probe; not a direct LibriSpeech paper comparison.",
        "runs": runs,
    }
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.manifest_out)


if __name__ == "__main__":
    main()
