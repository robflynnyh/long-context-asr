#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


HEAD_PRESETS = {
    "random_linear": {"head": "linear"},
    "random_bilstm": {"head": "bilstm"},
}


def parse_csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def lr_token(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("+", "")


def base_model_config():
    return {
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
    }


def base_config(args, head_name, learning_rate, run_label):
    checkpoint_path = Path(args.ssl_checkpoint)
    diagnostics_path = Path(args.run_dir) / "diagnostics" / f"{run_label}.jsonl"
    preset = HEAD_PRESETS[head_name]
    probe = {
        "ssl_checkpoint": str(checkpoint_path),
        "source_checkpoint": args.source_checkpoint,
        "head": preset["head"],
        "hidden_state_weighted_sum": True,
        "num_hidden_states": args.num_hidden_states,
        "load_decoder_from_ssl": False,
        "trainable_prefixes": ["decoder.", "weighted_sum."],
        "unfreeze_top_n_layers": 0,
        "trainable_encoder_layers": [],
        "encoder_lr_scale": 0.0,
    }
    if preset["head"] == "bilstm":
        probe.update(
            {
                "bilstm_hidden_size": args.bilstm_hidden_size,
                "bilstm_num_layers": args.bilstm_num_layers,
                "bilstm_dropout": args.bilstm_dropout,
            }
        )

    return {
        "model_class": "SCConformerXL",
        "description": (
            "ROB-129 fully frozen ROB-100 BEST-RQ CTC probe using corrected "
            "TEDLIUM STM utterance-boundary training examples."
        ),
        "model": base_model_config(),
        "probe": probe,
        "optimizer": {"name": "madgrad", "args": {"lr": learning_rate}},
        "scheduler": {"name": args.scheduler, "warmup_steps": args.warmup_steps},
        "audio_chunking": {"size": args.seq_len, "overlap": 0},
        "wandb": {
            "use": not args.disable_wandb and (not args.smoke or args.enable_smoke_wandb),
            "project_name": "rob129_frozen_rob100_probe",
            "name": f"{Path(args.run_dir).name}_{run_label}_frozen_weighted",
            "id": "",
            "dir": str(Path(args.run_dir) / "wandb"),
            "watch_model": False,
            "update_config_with_wandb_id": False,
        },
        "checkpointing": {
            "dir": str(Path(args.checkpoint_root) / run_label),
            "save_every_n_steps": args.save_every_n_steps,
        },
        "data": {
            "path": args.train_data_path,
            "format": "utterance_folder",
            "max_records": args.smoke_max_records if args.smoke else args.max_records,
        },
        "training": {
            "batch_size": args.smoke_batch_size if args.smoke else args.batch_size,
            "backprop_every": 1,
            "backwards_every": 1,
            "max_epochs": 1 if args.smoke else args.max_epochs,
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
    parser.add_argument("--train-data-path", required=True)
    parser.add_argument("--cache-contract", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument(
        "--ssl-checkpoint",
        default="/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-126/source-checkpoints/step_105360.pt",
    )
    parser.add_argument(
        "--source-checkpoint",
        default="/mnt/parscratch/users/acp21rjf/spotify/bestrq_ssl/rob100_papermask_p012_l4_sc_off_20260520/step_105360.pt",
    )
    parser.add_argument("--heads", default="random_linear,random_bilstm")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rates")
    parser.add_argument("--scheduler", choices=["cosine", "constant"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-n-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-max-records", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--enable-smoke-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--num-hidden-states", type=int, default=6)
    parser.add_argument("--bilstm-hidden-size", type=int, default=1024)
    parser.add_argument("--bilstm-num-layers", type=int, default=2)
    parser.add_argument("--bilstm-dropout", type=float, default=0.2)
    args = parser.parse_args()

    ssl_checkpoint = Path(args.ssl_checkpoint)
    if not ssl_checkpoint.is_file():
        raise SystemExit(f"missing local ROB-100 checkpoint: {ssl_checkpoint}")
    cache_contract = Path(args.cache_contract)
    if not cache_contract.is_file():
        raise SystemExit(f"missing cache contract summary: {cache_contract}")

    heads = parse_csv(args.heads) or ["random_bilstm"]
    unknown = [head for head in heads if head not in HEAD_PRESETS]
    if unknown:
        raise SystemExit(f"unknown head preset(s): {', '.join(unknown)}")
    learning_rates = parse_csv(args.learning_rates, float) or [args.learning_rate]
    single_lr = len(learning_rates) == 1

    config_dir = Path(args.run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)

    runs = []
    for head_name in heads:
        for learning_rate in learning_rates:
            run_label = head_name if single_lr else f"{head_name}_lr{lr_token(learning_rate)}"
            config = OmegaConf.create(base_config(args, head_name, learning_rate, run_label))
            config_path = config_dir / f"rob129_{run_label}_frozen_weighted_ctc_probe.yaml"
            OmegaConf.save(config=config, f=config_path)
            runs.append(
                {
                    "label": run_label,
                    "base_label": head_name,
                    "learning_rate": learning_rate,
                    "scheduler": args.scheduler,
                    "warmup_steps": args.warmup_steps,
                    "probe_head": HEAD_PRESETS[head_name]["head"],
                    "hidden_state_weighted_sum": True,
                    "num_hidden_states": args.num_hidden_states,
                    "hidden_state_exposure": (
                        "six SCConformerXL post-layer hidden states from layers 0-5; "
                        "weighted sum includes the final encoder layer state and excludes the pre-layer input"
                    ),
                    "unfreeze_top_n_layers": 0,
                    "trainable_encoder_layers": [],
                    "max_epochs": config.training.max_epochs,
                    "source_checkpoint": args.source_checkpoint,
                    "local_checkpoint": str(ssl_checkpoint),
                    "train_config": str(config_path),
                    "checkpoint_dir": str(Path(args.checkpoint_root) / run_label),
                    "diagnostics_path": config.training.diagnostics_path,
                }
            )

    manifest = {
        "issue": "ROB-129",
        "mode": "smoke" if args.smoke else "full",
        "run_dir": args.run_dir,
        "train_data_path": args.train_data_path,
        "train_data_format": "utterance_folder",
        "cache_contract": args.cache_contract,
        "checkpoint_root": args.checkpoint_root,
        "max_epochs": 1 if args.smoke else args.max_epochs,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.smoke_batch_size if args.smoke else args.batch_size,
        "comparison_note": (
            "TEDLIUM transfer evidence only. This reruns the fully frozen ROB-100 "
            "weighted hidden-state probe on corrected STM utterance-boundary labels."
        ),
        "runs": runs,
    }
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.manifest_out)


if __name__ == "__main__":
    main()
