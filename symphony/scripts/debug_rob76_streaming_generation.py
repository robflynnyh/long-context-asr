import argparse
import os
from types import SimpleNamespace
from typing import Optional

import lcasr
import torch
from omegaconf import OmegaConf

from exp.train_streaming_decoder_asr import decode_prediction_ids, make_dataloader
from lcasr.utils.audio_tools import total_frames
from lcasr.utils.general import find_latest_checkpoint, get_model_class, load_model
from lcasr.utils.streaming_targets import (
    build_streaming_frame_targets,
    filter_words_by_end_frame,
    pad_audio_for_streaming_delay,
)


def load_checkpoint_state(checkpoint_dir: str, checkpoint_name: Optional[str]):
    if checkpoint_name is None:
        checkpoint_name = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_name is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    path = os.path.join(checkpoint_dir, checkpoint_name)
    return path, torch.load(path, map_location="cpu")


def first_words(transcript, limit: int = 40) -> str:
    words = [str(item.get("word", item.get("text", ""))) for item in transcript]
    return " ".join(word for word in words if word)[:limit]


def main():
    parser = argparse.ArgumentParser(description="Compare ROB-76 teacher-forced and free-running generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--sample-runs", type=int, default=1)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config["training"]["batch_size"] = 1
    config["data"]["max_records"] = args.max_records
    config["wandb"]["use"] = False

    tokenizer_kwargs = {}
    if "tokenizer_path" in config["training"]:
        tokenizer_kwargs["tokenizer_path"] = config["training"]["tokenizer_path"]
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)

    model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
    checkpoint_path, checkpoint = load_checkpoint_state(args.checkpoint_dir, args.checkpoint_name)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model = model.to(device).eval()
    model_dtype = next(model.parameters()).dtype

    loader_args = SimpleNamespace(num_workers=0, pin_memory=False, prefetch_factor=1)
    dataloader = make_dataloader(config, tokenizer, loader_args, seen_ids=[])

    chunk_size = int(config["audio_chunking"]["size"])
    chunk_overlap = int(config["audio_chunking"].get("overlap", 0))
    stride = chunk_size - chunk_overlap
    delay_seconds = float(config["streaming"].get("delay_seconds", 2.0))
    buffer_seconds = float(config["streaming"].get("buffer_seconds", 0.25))
    silence_soft_dilation_seconds = float(config["streaming"].get("silence_soft_dilation_seconds", 0.0))
    silence_soft_dilation_direction = str(config["streaming"].get("silence_soft_dilation_direction", "past"))
    silence_soft_dilation_frames = int(round(total_frames(silence_soft_dilation_seconds) / model.subsampling_factor))
    silence_id = model.get_silence_id()
    max_frames = None if args.max_frames <= 0 else args.max_frames

    print(f"checkpoint={checkpoint_path}")
    print(f"step={checkpoint.get('podcast_step')} epoch={checkpoint.get('epoch')}")
    print(f"device={device} silence_id={silence_id} max_frames={max_frames}")
    prediction_head_type = getattr(model, "prediction_head_type", "two_head")
    print(f"prediction_head_type={prediction_head_type}")
    print(
        "silence_target_soft_dilation="
        f"{silence_soft_dilation_seconds:.3f}s {silence_soft_dilation_direction} "
        f"-> {silence_soft_dilation_frames} decoder frames"
    )
    if prediction_head_type == "two_head":
        print(
            "silence_head_sampling="
            f"runs={args.sample_runs} temperature={args.sample_temperature} "
            f"seed={args.sample_seed}; text head is greedy"
        )
    else:
        print(
            "single_head_sampling="
            f"runs={args.sample_runs} temperature={args.sample_temperature} "
            f"seed={args.sample_seed}; samples full vocab+silence distribution"
        )

    reported = 0
    with torch.no_grad():
        for audio, audio_lengths, transcripts, ids in dataloader:
            for chunk_start in range(0, int(audio_lengths.max().item()), stride):
                active = audio_lengths > chunk_start
                if active.sum().item() == 0:
                    continue
                chunk = audio[active, :, chunk_start : chunk_start + chunk_size]
                chunk_lengths = torch.clamp(audio_lengths[active] - chunk_start, min=0, max=chunk.size(-1))
                chunk_transcripts = [
                    filter_words_by_end_frame(transcripts[i], chunk_start, chunk_start + chunk_size)
                    for i, keep in enumerate(active.tolist())
                    if keep
                ]
                chunk_starts = torch.full((active.sum().item(),), chunk_start, dtype=torch.long)
                chunk, chunk_lengths = pad_audio_for_streaming_delay(
                    chunk,
                    chunk_lengths,
                    delay_seconds=delay_seconds,
                    buffer_seconds=buffer_seconds,
                )
                chunk = chunk.to(device=device, dtype=model_dtype)
                chunk_lengths = chunk_lengths.to(device)
                output_lengths = model.output_lengths(chunk_lengths)
                frame_targets = build_streaming_frame_targets(
                    transcripts=chunk_transcripts,
                    output_lengths=output_lengths.cpu(),
                    tokenizer=dataloader.tokenizer,
                    subsampling_factor=model.subsampling_factor,
                    delay_seconds=delay_seconds,
                    chunk_start_frames=chunk_starts,
                    silence_id=silence_id,
                ).to(device)
                valid = frame_targets != -100
                target_ns = ((frame_targets != silence_id) & valid).float().sum() / valid.float().sum().clamp_min(1)

                loss_out = model.calc_loss(
                    audio_signal=chunk,
                    length=chunk_lengths,
                    frame_targets=frame_targets,
                    silence_soft_dilation_frames=silence_soft_dilation_frames,
                    silence_soft_dilation_direction=silence_soft_dilation_direction,
                )
                tf_ids = loss_out["predictions"][0, : int(output_lengths[0].item())].detach().cpu().tolist()
                tf_text = decode_prediction_ids(tokenizer, tf_ids, silence_id=silence_id, max_tokens=80)

                silence_feedback_targets = torch.full_like(frame_targets, silence_id)
                silence_feedback_targets = silence_feedback_targets.masked_fill(~valid, -100)
                silence_feedback = model.forward(
                    audio_signal=chunk,
                    length=chunk_lengths,
                    frame_targets=silence_feedback_targets,
                    return_logits=True,
                )
                if prediction_head_type == "two_head":
                    silence_predictions = model._predict_ids(
                        silence_feedback["silence_logits"],
                        silence_feedback["text_logits"],
                    )
                else:
                    silence_predictions = silence_feedback["logits"].argmax(dim=-1)
                silence_ids = silence_predictions[0, : int(output_lengths[0].item())].detach().cpu().tolist()
                silence_fb_ns = sum(int(idx) != silence_id for idx in silence_ids) / max(len(silence_ids), 1)
                silence_fb_text = decode_prediction_ids(tokenizer, silence_ids, silence_id=silence_id, max_tokens=80)

                free = model.greedy_decode(audio_signal=chunk[:1], length=chunk_lengths[:1], max_frames=max_frames)
                free_len = int(free["length"][0].item())
                free_ids = free["predictions"][0, :free_len].detach().cpu().tolist()
                free_ns = sum(int(idx) != silence_id for idx in free_ids) / max(free_len, 1)
                free_text = decode_prediction_ids(tokenizer, free_ids, silence_id=silence_id, max_tokens=80)

                display = loss_out["display_losses"]
                active_ids = [ids[i] for i, keep in enumerate(active.tolist()) if keep]
                print(
                    "sample="
                    f"{reported} id={active_ids[0]} chunk_start={chunk_start} "
                    f"words={len(chunk_transcripts[0])} ref='{first_words(chunk_transcripts[0])}'"
                )
                print(
                    "  metrics "
                    f"target_ns={float(target_ns.cpu()):.4f} "
                    f"soft_target_ns={display['soft_non_silence_fraction']:.4f} "
                    f"teacher_forced_pred_ns={display['predicted_non_silence_fraction']:.4f} "
                    f"silence_feedback_pred_ns={silence_fb_ns:.4f} "
                    f"free_pred_ns={free_ns:.4f} loss={display['loss']:.4f}"
                )
                print(f"  teacher_forced_text='{tf_text[:200]}'")
                print(f"  silence_feedback_text='{silence_fb_text[:200]}'")
                print(f"  free_text='{free_text[:200]}'")
                for sample_run in range(args.sample_runs):
                    torch.manual_seed(args.sample_seed + reported * max(args.sample_runs, 1) + sample_run)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(args.sample_seed + reported * max(args.sample_runs, 1) + sample_run)
                    sampled = model.sample_decode(
                        audio_signal=chunk[:1],
                        length=chunk_lengths[:1],
                        max_frames=max_frames,
                        temperature=args.sample_temperature,
                    )
                    sampled_len = int(sampled["length"][0].item())
                    sampled_ids = sampled["predictions"][0, :sampled_len].detach().cpu().tolist()
                    sampled_ns = sum(int(idx) != silence_id for idx in sampled_ids) / max(sampled_len, 1)
                    sampled_text = decode_prediction_ids(
                        tokenizer,
                        sampled_ids,
                        silence_id=silence_id,
                        max_tokens=80,
                    )
                    print(
                        f"  sampled[{sample_run}]_pred_ns={sampled_ns:.4f} "
                        f"text='{sampled_text[:200]}'"
                    )

                reported += 1
                if reported >= args.samples:
                    return


if __name__ == "__main__":
    main()
