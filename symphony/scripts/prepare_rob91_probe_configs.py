#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


CHECKPOINTS = [
    ("25pct", "step_25344.pt"),
    ("50pct", "step_52800.pt"),
    ("100pct", "step_105360.pt"),
]


def base_config(args, label, checkpoint_file):
    batch_size = args.smoke_batch_size if args.smoke else args.batch_size
    max_epochs = 1 if args.smoke else args.max_epochs
    run_id = Path(args.run_dir).name
    return {
        "model_class": "SCConformerXL",
        "description": f"ROB-91 frozen BEST-RQ CTC probe for {label}.",
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
            "ssl_checkpoint": str(Path(args.checkpoint_cache) / checkpoint_file),
            "source_checkpoint": f"{args.stanage_checkpoint_dir}/{checkpoint_file}",
            "load_decoder_from_ssl": False,
            "trainable_prefixes": ["decoder."],
        },
        "optimizer": {"name": "madgrad", "args": {"lr": args.learning_rate}},
        "scheduler": {"warmup_steps": args.warmup_steps},
        "audio_chunking": {"size": args.seq_len, "overlap": 0},
        "wandb": {
            "use": not args.disable_wandb and (not args.smoke or args.enable_smoke_wandb),
            "project_name": "rob91_bestrq_ctc_probe",
            "name": f"{run_id}_{label}_frozen_ctc_probe",
            "id": "",
            "dir": str(Path(args.run_dir) / "wandb"),
            "update_config_with_wandb_id": False,
        },
        "checkpointing": {
            "dir": str(Path(args.checkpoint_root) / label),
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
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-cache", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--stanage-checkpoint-dir", default="/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/6l_2048_1epoch_lr3e4_20260516")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable-smoke-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-every-n-steps", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    labels = CHECKPOINTS[:1] if args.smoke else CHECKPOINTS
    config_dir = Path(args.run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)

    runs = []
    for label, checkpoint_file in labels:
        config = OmegaConf.create(base_config(args, label, checkpoint_file))
        config_path = config_dir / f"rob91_{label}_frozen_ctc_probe.yaml"
        OmegaConf.save(config=config, f=config_path)
        runs.append(
            {
                "label": label,
                "checkpoint_file": checkpoint_file,
                "source_checkpoint": f"{args.stanage_checkpoint_dir}/{checkpoint_file}",
                "local_checkpoint": str(Path(args.checkpoint_cache) / checkpoint_file),
                "train_config": str(config_path),
                "checkpoint_dir": str(Path(args.checkpoint_root) / label),
            }
        )

    manifest = {
        "mode": "smoke" if args.smoke else "full",
        "run_dir": args.run_dir,
        "train_manifest": args.train_manifest,
        "checkpoint_cache": args.checkpoint_cache,
        "checkpoint_root": args.checkpoint_root,
        "runs": runs,
    }
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.manifest_out)


if __name__ == "__main__":
    main()
