#!/usr/bin/env python3
import argparse
import json
import os

import torch
from omegaconf import OmegaConf


def remap_pair_paths(record):
    audio_name = os.path.basename(record["audio"])
    text_name = os.path.basename(record["txt"])
    return {
        **record,
        "audio": os.path.join("/store/store5/data/spotify/spotify_10percent/audio", audio_name),
        "txt": os.path.join("/store/store5/data/spotify/spotify_10percent/text", text_name),
    }


def load_pairs(path):
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)


def validate_paths(pairs):
    missing = []
    for key, record in pairs.items():
        if not os.path.exists(record["audio"]):
            missing.append((key, "audio", record["audio"]))
        if not os.path.exists(record["txt"]):
            missing.append((key, "txt", record["txt"]))
        if len(missing) >= 10:
            break
    if missing:
        detail = "\n".join(f"{key} {kind}: {path}" for key, kind, path in missing)
        raise FileNotFoundError(f"missing remapped Spotify files:\n{detail}")


def prepare_seed_checkpoint(seed_checkpoint, checkpoint_dir, config, force):
    if not os.path.exists(seed_checkpoint):
        raise FileNotFoundError(f"seed checkpoint does not exist: {seed_checkpoint}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    seed_out = os.path.join(checkpoint_dir, "step_0.pt")
    if os.path.exists(seed_out) and not force:
        print(f"seed_checkpoint_out={seed_out}")
        print("seed_checkpoint_status=exists")
        return seed_out

    checkpoint = torch.load(seed_checkpoint, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"seed checkpoint has no model state: {seed_checkpoint}")
    torch.save(
        {
            "model": checkpoint["model"],
            "podcast_step": 0,
            "config": config,
            "seen_ids": [],
            "epoch": 0,
            "source_checkpoint": seed_checkpoint,
            "source_podcast_step": checkpoint.get("podcast_step"),
            "source_epoch": checkpoint.get("epoch"),
        },
        seed_out,
    )
    print(f"seed_checkpoint_out={seed_out}")
    print("seed_checkpoint_status=written")
    print(f"source_checkpoint={seed_checkpoint}")
    print(f"source_podcast_step={checkpoint.get('podcast_step')}")
    print(f"source_epoch={checkpoint.get('epoch')}")
    return seed_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="exp/configs/streaming_decoder_asr_100m.yaml")
    parser.add_argument("--source-pairs", default="/store/store5/data/spotify/renamed_audio_text_pairs_10_percent.json")
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--seed-checkpoint",
        default="/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_two_head_rob92_mimas_5epoch_rob92-mimas-5epoch-continuation-20260518T221140Z/step_137895.pt",
    )
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--wandb-name", default="rob105_streaming_decoder_asr_rl_grpo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-frames", type=int, default=96)
    parser.add_argument("--reward-std-min", type=float, default=0.01)
    parser.add_argument("--sample-text-log-every", type=int, default=10)
    parser.add_argument("--reward-wer-weight", type=float, default=0.7)
    parser.add_argument("--reward-cer-weight", type=float, default=0.3)
    parser.add_argument("--force-seed", action="store_true")
    parser.add_argument("--no-validate-paths", action="store_true")
    args = parser.parse_args()

    pairs = {key: remap_pair_paths(record) for key, record in load_pairs(args.source_pairs).items()}
    if not args.no_validate_paths:
        validate_paths(pairs)
    write_json(args.manifest_out, pairs)

    config = OmegaConf.load(args.base_config)
    config.data.path = args.manifest_out
    if "max_records" in config.data:
        del config.data.max_records
    config.checkpointing.dir = args.checkpoint_dir
    config.checkpointing.save_every_n_steps = args.save_every
    config.training.batch_size = args.batch_size
    config.training.max_steps = args.max_steps
    config.optimizer.args.lr = args.learning_rate
    config.scheduler.name = "constant"
    config.wandb.use = True
    config.wandb.name = args.wandb_name
    config.wandb.id = ""
    config.wandb.dir = args.wandb_dir
    config.wandb.update_config_with_wandb_id = False
    config.rl = {
        "algorithm": "grpo",
        "num_rollouts": args.num_rollouts,
        "temperature": args.temperature,
        "max_output_frames": args.max_output_frames,
        "max_decode_tokens": 256,
        "reward_type": "weighted_error",
        "reward_wer_weight": args.reward_wer_weight,
        "reward_cer_weight": args.reward_cer_weight,
        "reward_offset": 1.0,
        "reward_scale": 1.0,
        "reward_min": 0.0,
        "reward_max": None,
        "reward_positive_threshold": None,
        "reward_std_min": args.reward_std_min,
        "advantage_eps": 1e-6,
        "shuffle_chunks": True,
        "include_empty_references": False,
        "sample_text_log_every": args.sample_text_log_every,
    }

    os.makedirs(os.path.dirname(args.config_out), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.wandb_dir, exist_ok=True)
    prepare_seed_checkpoint(args.seed_checkpoint, args.checkpoint_dir, config, args.force_seed)
    OmegaConf.save(config=config, f=args.config_out)

    durations = [float(record.get("duration", 0.0)) for record in pairs.values()]
    print(f"records={len(pairs)}")
    print(f"total_duration_hours={sum(durations) / 3600.0:.2f}")
    print(f"manifest={args.manifest_out}")
    print(f"config={args.config_out}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"seed_checkpoint={args.seed_checkpoint}")
    print(f"wandb_dir={args.wandb_dir}")
    print(f"wandb_name={args.wandb_name}")
    print(f"batch_size={args.batch_size}")
    print(f"max_steps={args.max_steps}")
    print(f"save_every={args.save_every}")
    print(f"learning_rate={args.learning_rate}")
    print(f"num_rollouts={args.num_rollouts}")
    print(f"temperature={args.temperature}")
    print(f"max_output_frames={args.max_output_frames}")
    print(f"reward_std_min={args.reward_std_min}")
    print(f"sample_text_log_every={args.sample_text_log_every}")


if __name__ == "__main__":
    main()
