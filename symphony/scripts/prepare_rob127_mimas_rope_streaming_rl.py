#!/usr/bin/env python3
import argparse
import os

from omegaconf import OmegaConf

from prepare_rob105_mimas_streaming_rl import (
    load_pairs,
    prepare_seed_checkpoint,
    remap_pair_paths,
    validate_paths,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="exp/configs/streaming_decoder_asr_100m.yaml")
    parser.add_argument("--source-pairs", default="/store/store5/data/spotify/renamed_audio_text_pairs_10_percent.json")
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--seed-checkpoint",
        default="/store/store5/data/acp21rjf/spotify/streaming_decoder_asr_100m_rope_rob116_mimas_5epoch_rob116-rope-mimas-5epoch-20260521T153748Z/step_98629.pt",
    )
    parser.add_argument("--wandb-dir", required=True)
    parser.add_argument("--wandb-name", default="rob127_rope_streaming_decoder_asr_rl_grpo")
    parser.add_argument("--wandb-id", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-rollouts", type=int, default=6)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-frames", type=int, default=None)
    parser.add_argument("--reward-std-min", type=float, default=0.01)
    parser.add_argument("--sample-text-log-every", type=int, default=10)
    parser.add_argument("--reward-wer-weight", type=float, default=0.7)
    parser.add_argument("--reward-cer-weight", type=float, default=0.3)
    parser.add_argument("--late-word-tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--late-word-penalty-per-second", type=float, default=0.25)
    parser.add_argument("--late-word-penalty-max", type=float, default=1.0)
    parser.add_argument("--rotary-base-freq", type=int, default=1_500_000)
    parser.add_argument("--rotary-interpolation-factor", type=float, default=1.0)
    parser.add_argument("--force-seed", action="store_true")
    parser.add_argument("--no-validate-paths", action="store_true")
    args = parser.parse_args()

    pairs = {key: remap_pair_paths(record) for key, record in load_pairs(args.source_pairs).items()}
    if not args.no_validate_paths:
        validate_paths(pairs)
    write_json(args.manifest_out, pairs)

    config = OmegaConf.load(args.base_config)
    config.model.use_rotary = True
    config.model.rotary_base_freq = args.rotary_base_freq
    config.model.rotary_interpolation_factor = args.rotary_interpolation_factor
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
    config.wandb.id = args.wandb_id
    config.wandb.dir = args.wandb_dir
    config.wandb.update_config_with_wandb_id = False
    config.rl = {
        "algorithm": "grpo",
        "num_rollouts": args.num_rollouts,
        "microbatch_size": args.microbatch_size,
        "temperature": args.temperature,
        "max_output_frames": args.max_output_frames,
        "reward_type": "weighted_error",
        "reward_wer_weight": args.reward_wer_weight,
        "reward_cer_weight": args.reward_cer_weight,
        "late_word_tolerance_seconds": args.late_word_tolerance_seconds,
        "late_word_penalty_per_second": args.late_word_penalty_per_second,
        "late_word_penalty_max": args.late_word_penalty_max,
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
    print(f"wandb_id={args.wandb_id or 'new'}")
    print(f"batch_size={args.batch_size}")
    print(f"max_steps={args.max_steps}")
    print(f"save_every={args.save_every}")
    print(f"learning_rate={args.learning_rate}")
    print(f"num_rollouts={args.num_rollouts}")
    print(f"microbatch_size={args.microbatch_size}")
    print(f"temperature={args.temperature}")
    print(f"max_output_frames={args.max_output_frames if args.max_output_frames is not None else 'uncapped'}")
    print(f"reward_std_min={args.reward_std_min}")
    print(f"sample_text_log_every={args.sample_text_log_every}")
    print(f"late_word_tolerance_seconds={args.late_word_tolerance_seconds}")
    print(f"late_word_penalty_per_second={args.late_word_penalty_per_second}")
    print(f"late_word_penalty_max={args.late_word_penalty_max}")
    print(f"use_rotary={config.model.use_rotary}")
    print(f"rotary_base_freq={config.model.rotary_base_freq}")
    print(f"rotary_interpolation_factor={config.model.rotary_interpolation_factor}")


if __name__ == "__main__":
    main()
