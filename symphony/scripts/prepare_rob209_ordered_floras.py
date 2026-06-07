#!/usr/bin/env python3
import argparse
import json
import os

from omegaconf import OmegaConf

from lcasr.utils.streaming_targets import streaming_padding_frames


ROB123_PRETRAINED_CHECKPOINT = (
    "/mnt/parscratch/users/acp21rjf/spotify/"
    "streaming_decoder_asr_100m_rope_rob123_full_spotify_2epoch_delay0p5_"
    "rob123-rope-full-spotify-2epoch-delay0p5-20260523T091306Z/step_272362.pt"
)


def load_pairs(path):
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def validate_paths(pairs, limit):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="exp/configs/streaming_decoder_asr_100m.yaml")
    parser.add_argument(
        "--source-pairs",
        default="/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-81/manifests/floras50_safe_norm_drop_oov.json",
    )
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--pretrained-checkpoint", default=ROB123_PRETRAINED_CHECKPOINT)
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--wandb-name", default="rob209_ordered_floras_bounded_history")
    parser.add_argument("--batch-size", type=int, default=88)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-every-n-steps", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--subsampling-history-frames", type=int, default=2048)
    parser.add_argument("--decoder-history-frames", type=int, default=2048)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--debug-generate-every-records", type=int, default=500)
    parser.add_argument("--debug-generate-max-frames", type=int, default=0)
    parser.add_argument("--random-seed", default="6433")
    parser.add_argument("--validate-path-limit", type=int, default=20)
    parser.add_argument("--validate-all-paths", action="store_true")
    parser.add_argument("--no-validate-paths", action="store_true")
    args = parser.parse_args()

    pairs = load_pairs(args.source_pairs)
    validate_limit = None if args.validate_all_paths else args.validate_path_limit
    checked_paths = 0
    if not args.no_validate_paths:
        checked_paths = validate_paths(pairs, validate_limit)
    if args.pretrained_checkpoint and not os.path.exists(args.pretrained_checkpoint):
        raise FileNotFoundError(f"pretrained checkpoint does not exist: {args.pretrained_checkpoint}")

    config = OmegaConf.load(args.base_config)
    config.data.path = args.source_pairs
    if "max_records" in config.data:
        del config.data.max_records
    config.checkpointing.dir = args.checkpoint_dir
    config.checkpointing.pretrained = args.pretrained_checkpoint
    config.checkpointing.save_every_n_steps = args.save_every_n_steps
    config.audio_chunking.size = args.chunk_size
    config.audio_chunking.overlap = 0
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.shuffle_chunks = False
    config.training.random_seed = args.random_seed if args.random_seed == "random" else int(args.random_seed)
    config.training.ordered_chunk_training = {
        "enabled": True,
        "subsampling_history_frames": args.subsampling_history_frames,
        "decoder_history_frames": args.decoder_history_frames,
        "detach_cache": True,
    }
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
    config.wandb.name = args.wandb_name
    config.wandb.id = ""
    config.wandb.dir = args.wandb_dir
    config.wandb.update_config_with_wandb_id = False

    os.makedirs(os.path.dirname(args.config_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.wandb_dir, exist_ok=True)
    OmegaConf.save(config=config, f=args.config_out)

    durations = [float(record.get("duration", 0.0)) for record in pairs.values()]
    total_hours = sum(durations) / 3600.0
    final_flush_frames = streaming_padding_frames(config.streaming.delay_seconds, config.streaming.buffer_seconds)
    lines = [
        f"records={len(pairs)}",
        f"total_duration_hours={total_hours:.2f}",
        f"source_pairs={args.source_pairs}",
        f"validated_path_records={checked_paths}",
        f"runtime_config={args.config_out}",
        f"checkpoint_dir={args.checkpoint_dir}",
        f"pretrained_checkpoint={args.pretrained_checkpoint}",
        f"wandb_dir={args.wandb_dir}",
        f"wandb_name={args.wandb_name}",
        f"batch_size={config.training.batch_size}",
        f"max_epochs={config.training.max_epochs}",
        f"learning_rate={config.optimizer.args.lr}",
        f"save_every_n_steps={config.checkpointing.save_every_n_steps}",
        f"chunk_size={config.audio_chunking.size}",
        f"shuffle_chunks={config.training.shuffle_chunks}",
        f"ordered_chunk_training={config.training.ordered_chunk_training.enabled}",
        f"subsampling_history_frames={config.training.ordered_chunk_training.subsampling_history_frames}",
        f"decoder_history_frames={config.training.ordered_chunk_training.decoder_history_frames}",
        f"detach_cache={config.training.ordered_chunk_training.detach_cache}",
        f"use_rotary={config.model.use_rotary}",
        f"rotary_base_freq={config.model.rotary_base_freq}",
        f"rotary_interpolation_factor={config.model.rotary_interpolation_factor}",
        f"delay_seconds={config.streaming.delay_seconds}",
        f"buffer_seconds={config.streaming.buffer_seconds}",
        f"final_flush_frames={final_flush_frames}",
        f"debug_generate_every_records={args.debug_generate_every_records}",
        f"debug_generate_max_frames={args.debug_generate_max_frames}",
    ]
    with open(args.summary_out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
