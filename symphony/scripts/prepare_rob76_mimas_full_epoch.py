#!/usr/bin/env python3
import argparse
import json
import os

from omegaconf import OmegaConf


def remap_pair_paths(record):
    audio_name = os.path.basename(record["audio"])
    text_name = os.path.basename(record["txt"])
    return {
        **record,
        "audio": os.path.join(
            "/store/store5/data/spotify/spotify_10percent/audio",
            audio_name,
        ),
        "txt": os.path.join(
            "/store/store5/data/spotify/spotify_10percent/text",
            text_name,
        ),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="exp/configs/streaming_decoder_asr_100m.yaml")
    parser.add_argument(
        "--source-pairs",
        default="/store/store5/data/spotify/renamed_audio_text_pairs_10_percent.json",
    )
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--wandb-name", default="streaming_decoder_asr_100m_mimas_full_epoch")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--debug-generate-every-records", type=int, default=0)
    parser.add_argument("--debug-generate-max-frames", type=int, default=0)
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
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    if args.learning_rate is not None:
        config.optimizer.args.lr = args.learning_rate
    if args.debug_generate_every_records > 0:
        config.training.debug_generation = {
            "enabled": True,
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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.wandb_dir, exist_ok=True)
    OmegaConf.save(config=config, f=args.config_out)

    durations = [float(record.get("duration", 0.0)) for record in pairs.values()]
    total_hours = sum(durations) / 3600.0
    print(f"records={len(pairs)}")
    print(f"total_duration_hours={total_hours:.2f}")
    print(f"manifest={args.manifest_out}")
    print(f"config={args.config_out}")
    print(f"checkpoint_dir={args.checkpoint_dir}")
    print(f"wandb_dir={args.wandb_dir}")
    print(f"wandb_name={args.wandb_name}")
    print(f"batch_size={args.batch_size}")
    print(f"max_epochs={args.max_epochs}")
    print(f"learning_rate={config.optimizer.args.lr}")
    if args.debug_generate_every_records > 0:
        print(f"debug_generate_every_records={args.debug_generate_every_records}")
        print(f"debug_generate_max_frames={args.debug_generate_max_frames}")


if __name__ == "__main__":
    main()
