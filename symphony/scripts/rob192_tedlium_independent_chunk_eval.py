#!/usr/bin/env python3
"""ROB-192 TEDLIUM diagnostic: decode fixed spectrogram chunks independently."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

import lcasr
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.general import get_model_class, load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval" / "tedlium"))
import run as tedlium_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode TEDLIUM recordings as independent fixed-size spectrogram chunks."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tedlium-root", default=os.environ.get("LCASR_TEDLIUM_ROOT", ""))
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--recording-index", type=int, default=None)
    parser.add_argument("--decode-mode", default="greedy", choices=["greedy", "sample", "sample_silence_greedy_text"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-frames", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--use-kv-cache", action="store_true")
    parser.add_argument("--max-kv-cache-length", type=int, default=None)
    parser.add_argument("--max-kv-cache-spectrogram-length", type=int, default=None)
    parser.add_argument("--eval-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def cuda_dtype(name, device):
    if device.type != "cuda" or name == "float32":
        return None
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def normalize(text):
    text = tedlium_run.normalize(text).lower().strip()
    return text[:-1].strip() if text.endswith(".") else text


def chunk_ranges(num_frames, chunk_size, chunk_overlap):
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be in [0, chunk_size)")
    step = chunk_size - chunk_overlap
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        yield start, end
        if end == num_frames:
            break
        start += step


def load_streaming_model(checkpoint_path, eval_dtype):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(config, tokenizer.vocab_size(), model_class=get_model_class(config=config))
    model.load_state_dict(checkpoint["model"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = cuda_dtype(eval_dtype, device)
    model = model.to(device=device, dtype=dtype) if dtype is not None else model.to(device=device)
    model.device = device
    model.eval()
    return model, tokenizer, device, dtype


def selected_recordings(data, recording_index, max_recordings):
    indexed = list(enumerate(data))
    if recording_index is not None:
        if recording_index < 0 or recording_index >= len(data):
            raise IndexError(f"recording-index {recording_index} out of range for {len(data)} recordings")
        indexed = [(recording_index, data[recording_index])]
    if max_recordings is not None:
        indexed = indexed[:max_recordings]
    return indexed


def main():
    args = parse_args()
    tedlium_run.TEST_PATH = str(Path(args.tedlium_root) / "test")
    tedlium_run.DEV_PATH = str(Path(args.tedlium_root) / "dev")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rob192_tedlium_independent_chunk_eval.csv"
    chunk_path = output_dir / "chunk_predictions.jsonl"
    summary_path = output_dir / "summary.json"

    if not args.tedlium_root:
        raise SystemExit("--tedlium-root is required when LCASR_TEDLIUM_ROOT is unset")

    model, tokenizer, device, dtype = load_streaming_model(args.checkpoint, args.eval_dtype)
    data = tedlium_run.get_text_and_audio(args.split)
    records = selected_recordings(data, args.recording_index, args.max_recordings)

    rows = []
    all_predictions = []
    all_references = []
    with chunk_path.open("w", encoding="utf-8") as chunk_handle:
        for recording_index, record in records:
            audio_spec, gold_text = record["process_fn"](record)
            gold_text = normalize(gold_text)
            chunk_predictions = []
            chunk_meta = []
            for chunk_index, (start, end) in enumerate(chunk_ranges(audio_spec.shape[-1], args.chunk_size, args.chunk_overlap)):
                chunk = audio_spec[:, start:end]
                original_frames = int(chunk.shape[-1])
                if original_frames < args.chunk_size:
                    chunk = torch.nn.functional.pad(chunk, (0, args.chunk_size - original_frames))
                result = model.transcribe(
                    chunk,
                    tokenizer,
                    device=device,
                    decode_mode=args.decode_mode,
                    temperature=args.temperature,
                    max_output_frames=args.max_output_frames,
                    max_tokens=args.max_tokens,
                    use_kv_cache=args.use_kv_cache,
                    max_kv_cache_length=args.max_kv_cache_length,
                    max_kv_cache_spectrogram_length=args.max_kv_cache_spectrogram_length,
                    return_metadata=True,
                )
                prediction = normalize(result["text"])
                chunk_predictions.append(prediction)
                item = {
                    "recording": record["id"],
                    "recording_index": recording_index,
                    "chunk_index": chunk_index,
                    "start_frame": start,
                    "end_frame": end,
                    "frames": original_frames,
                    "decode_frames": int(chunk.shape[-1]),
                    "prediction": prediction,
                    "output_frames": result["output_frames"],
                    "pred_non_silence_fraction": result["pred_non_silence_fraction"],
                }
                chunk_meta.append(item)
                chunk_handle.write(json.dumps(item, ensure_ascii=True) + "\n")
                if args.verbose:
                    print(json.dumps(item, ensure_ascii=True))

            prediction_text = " ".join(text for text in chunk_predictions if text).strip()
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
                hypotheses=[prediction_text],
                references=[gold_text],
            )
            rows.append(
                {
                    "dataset": "tedlium",
                    "split": args.split,
                    "recording": record["id"],
                    "recording_index": recording_index,
                    "wer": wer,
                    "words": words,
                    "ins_rate": ins_rate,
                    "del_rate": del_rate,
                    "sub_rate": sub_rate,
                    "hyp_words": len(prediction_text.split()),
                    "ref_words": len(gold_text.split()),
                    "chunks": len(chunk_meta),
                    "frames": int(audio_spec.shape[-1]),
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "decode_mode": args.decode_mode,
                    "use_kv_cache": args.use_kv_cache,
                    "eval_dtype": str(dtype).replace("torch.", "") if dtype is not None else "float32",
                    "device": str(device),
                }
            )
            all_predictions.append(prediction_text)
            all_references.append(gold_text)

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
        hypotheses=all_predictions,
        references=all_references,
    )
    rows.append(
        {
            "dataset": "tedlium",
            "split": args.split,
            "recording": "all",
            "recording_index": "",
            "wer": wer,
            "words": words,
            "ins_rate": ins_rate,
            "del_rate": del_rate,
            "sub_rate": sub_rate,
            "hyp_words": sum(len(text.split()) for text in all_predictions),
            "ref_words": sum(len(text.split()) for text in all_references),
            "chunks": sum(int(row["chunks"]) for row in rows),
            "frames": sum(int(row["frames"]) for row in rows),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "decode_mode": args.decode_mode,
            "use_kv_cache": args.use_kv_cache,
            "eval_dtype": str(dtype).replace("torch.", "") if dtype is not None else "float32",
            "device": str(device),
        }
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": args.checkpoint,
        "tedlium_root": args.tedlium_root,
        "split": args.split,
        "recordings": len(records),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "decode_mode": args.decode_mode,
        "use_kv_cache": args.use_kv_cache,
        "max_kv_cache_length": args.max_kv_cache_length,
        "max_kv_cache_spectrogram_length": args.max_kv_cache_spectrogram_length,
        "device": str(device),
        "eval_dtype": str(dtype).replace("torch.", "") if dtype is not None else "float32",
        "csv": str(csv_path),
        "chunk_predictions": str(chunk_path),
        "aggregate": rows[-1],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
