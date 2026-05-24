#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


HEAD_PRESETS = {
    "random_bilstm": {"head": "bilstm", "init_source": False},
    "random_linear": {"head": "linear", "init_source": False},
    "source_linear_trainable": {"head": "linear", "init_source": True},
}


def parse_csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def load_checkpoint_config(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        raise SystemExit(f"{path} has no config")
    config = OmegaConf.to_container(checkpoint["config"], resolve=True)
    if "model" not in config:
        raise SystemExit(f"{path} config has no model section")
    return config


def base_config(args, source_config, head_name):
    preset = HEAD_PRESETS[head_name]
    run_id = Path(args.run_dir).name
    diagnostics_path = Path(args.run_dir) / "diagnostics" / f"{head_name}.jsonl"
    model_config = dict(source_config["model"])
    model_config.setdefault("decoder_norm", True)

    probe = {
        "ssl_checkpoint": args.supervised_checkpoint,
        "source_checkpoint": args.supervised_checkpoint,
        "head": preset["head"],
        "load_decoder_from_ssl": True,
        "init_decoder_from_backbone_ctc": preset["init_source"],
        "trainable_prefixes": ["decoder."],
    }
    if args.hidden_state_weighted_sum:
        probe.update(
            {
                "hidden_state_weighted_sum": True,
                "num_hidden_states": args.num_hidden_states,
                "trainable_prefixes": ["decoder.", "weighted_sum."],
            }
        )
    if preset["head"] == "bilstm":
        probe.update(
            {
                "bilstm_hidden_size": args.bilstm_hidden_size,
                "bilstm_num_layers": args.bilstm_num_layers,
                "bilstm_dropout": args.bilstm_dropout,
            }
        )

    return {
        "model_class": source_config.get("model_class", "EncDecSconformerV2"),
        "description": (
            f"ROB-128 {head_name} CTC probe on frozen ROB-81 supervised encoder "
            "using TEDLIUM STM utterance boundaries."
        ),
        "model": model_config,
        "probe": probe,
        "optimizer": {"name": "madgrad", "args": {"lr": args.learning_rate}},
        "scheduler": {"name": args.scheduler, "warmup_steps": args.warmup_steps},
        "audio_chunking": {"size": args.seq_len, "overlap": 0},
        "wandb": {
            "use": not args.disable_wandb and (not args.smoke or args.enable_smoke_wandb),
            "project_name": "rob128_supervised_probe_debug",
            "name": f"{run_id}_{head_name}",
            "id": "",
            "dir": str(Path(args.run_dir) / "wandb"),
            "watch_model": False,
            "update_config_with_wandb_id": False,
        },
        "checkpointing": {
            "dir": str(Path(args.checkpoint_root) / head_name),
            "save_every_n_steps": args.save_every_n_steps,
        },
        "data": {"path": args.train_data_path, "format": "utterance_folder"},
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
    parser.add_argument("--utterance-summary", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument(
        "--supervised-checkpoint",
        default="/store/store5/data/acp21rjf/symphony-job-artifacts/ROB-81/checkpoints/supervised_floras50_spotifytok_safe_norm_drop_oov_lr1e-4_12ep_nw0/step_323484.pt",
    )
    parser.add_argument("--heads", default="random_bilstm")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable-smoke-wandb", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["cosine", "constant"], default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-n-steps", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--hidden-state-weighted-sum", action="store_true")
    parser.add_argument("--num-hidden-states", type=int, default=3)
    parser.add_argument("--bilstm-hidden-size", type=int, default=1024)
    parser.add_argument("--bilstm-num-layers", type=int, default=2)
    parser.add_argument("--bilstm-dropout", type=float, default=0.2)
    parser.add_argument("--stage", default="overfit")
    args = parser.parse_args()

    supervised_checkpoint = Path(args.supervised_checkpoint)
    if not supervised_checkpoint.exists():
        raise SystemExit(f"missing supervised checkpoint: {supervised_checkpoint}")
    source_config = load_checkpoint_config(supervised_checkpoint)
    heads = parse_csv(args.heads) or ["random_bilstm"]
    unknown = [head for head in heads if head not in HEAD_PRESETS]
    if unknown:
        raise SystemExit(f"unknown head preset(s): {', '.join(unknown)}")

    config_dir = Path(args.run_dir) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)

    runs = []
    for head_name in heads:
        config = OmegaConf.create(base_config(args, source_config, head_name))
        config_path = config_dir / f"rob128_{head_name}_ctc_probe.yaml"
        OmegaConf.save(config=config, f=config_path)
        runs.append(
            {
                "label": head_name,
                "learning_rate": args.learning_rate,
                "scheduler": args.scheduler,
                "warmup_steps": args.warmup_steps,
                "probe_head": HEAD_PRESETS[head_name]["head"],
                "init_decoder_from_backbone_ctc": HEAD_PRESETS[head_name]["init_source"],
                "hidden_state_weighted_sum": args.hidden_state_weighted_sum,
                "num_hidden_states": args.num_hidden_states if args.hidden_state_weighted_sum else None,
                "max_epochs": config.training.max_epochs,
                "source_checkpoint": args.supervised_checkpoint,
                "local_checkpoint": args.supervised_checkpoint,
                "train_config": str(config_path),
                "checkpoint_dir": str(Path(args.checkpoint_root) / head_name),
                "diagnostics_path": config.training.diagnostics_path,
            }
        )

    manifest = {
        "issue": "ROB-128",
        "stage": args.stage,
        "mode": "smoke" if args.smoke else "full",
        "run_dir": args.run_dir,
        "train_data_path": args.train_data_path,
        "train_data_format": "utterance_folder",
        "utterance_summary": args.utterance_summary,
        "checkpoint_root": args.checkpoint_root,
        "max_epochs": 1 if args.smoke else args.max_epochs,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.smoke_batch_size if args.smoke else args.batch_size,
        "supervised_checkpoint": args.supervised_checkpoint,
        "comparison_note": "Known-good ROB-81 supervised encoder; TEDLIUM train split uses STM utterance boundaries, not full-record chunking.",
        "runs": runs,
    }
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.manifest_out)


if __name__ == "__main__":
    main()
