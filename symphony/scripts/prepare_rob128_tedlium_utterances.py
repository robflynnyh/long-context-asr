#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import lcasr
import torch
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from lcasr.utils.audio_tools import processing_chain, total_frames


english_normalizer = EnglishTextNormalizer()


def normalize_stm_text(text: str) -> str:
    text = re.sub(r" '([a-z])", r"'\1", text)
    text = english_normalizer(text).lower().strip()
    return re.sub(r" +", " ", text).strip()


def read_stm_utterances(stm_path: Path):
    utterances = []
    for line in stm_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        start, end = float(parts[3]), float(parts[4])
        text = " ".join(parts[6:])
        if text == "ignore_time_segment_in_scoring":
            continue
        text = normalize_stm_text(text)
        if text:
            utterances.append({"start": start, "end": end, "text": text})
    return utterances


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ROB-128 TEDLIUM STM-boundary utterance examples for exp/train.py."
    )
    parser.add_argument("--tedlium-root", default="/store/store4/data/TEDLIUM_release1/legacy")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--recording-stem")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_root = Path(args.tedlium_root) / args.split
    sph_dir = split_root / "sph"
    stm_dir = split_root / "stm"
    if not sph_dir.exists():
        raise SystemExit(f"missing sph directory: {sph_dir}")
    if not stm_dir.exists():
        raise SystemExit(f"missing stm directory: {stm_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    sph_paths = sorted(sph_dir.glob("*.sph"))
    if args.recording_stem:
        sph_paths = [path for path in sph_paths if path.stem == args.recording_stem]
        if not sph_paths:
            raise SystemExit(f"recording not found in {sph_dir}: {args.recording_stem}")
    if args.max_recordings is not None:
        sph_paths = sph_paths[: args.max_recordings]

    saved = 0
    skipped_empty = 0
    recordings = []
    for sph_path in tqdm(sph_paths, desc=f"Preparing TEDLIUM {args.split} utterances"):
        if args.max_utterances is not None and saved >= args.max_utterances:
            break
        stem = sph_path.stem
        stm_path = stm_dir / f"{stem}.stm"
        if not stm_path.exists():
            raise SystemExit(f"missing STM for {sph_path}: {stm_path}")

        spec = processing_chain(str(sph_path))
        rec_saved = 0
        for utt_idx, utterance in enumerate(read_stm_utterances(stm_path)):
            if args.max_utterances is not None and saved >= args.max_utterances:
                break
            start_frame = total_frames(utterance["start"])
            end_frame = total_frames(utterance["end"])
            utterance_spec = spec[:, :, start_frame:end_frame]
            token_ids = tokenizer.encode(utterance["text"])
            if utterance_spec.shape[-1] == 0 or len(token_ids) == 0:
                skipped_empty += 1
                continue

            utt_id = f"{stem}_{utt_idx:05d}"
            out_path = output_dir / f"{utt_id}.pt"
            if args.force or not out_path.exists():
                torch.save(
                    {
                        "id": utt_id,
                        "recording": stem,
                        "start": utterance["start"],
                        "end": utterance["end"],
                        "text": utterance["text"],
                        "audio": utterance_spec,
                        "txt": torch.LongTensor(token_ids).unsqueeze(0),
                        "txt_lengths": torch.LongTensor([len(token_ids)]),
                        "audio_lengths": torch.LongTensor([utterance_spec.shape[-1]]),
                    },
                    out_path,
                )
            saved += 1
            rec_saved += 1
        recordings.append({"recording": stem, "utterances": rec_saved})

    summary = {
        "issue": "ROB-128",
        "split": args.split,
        "tedlium_root": args.tedlium_root,
        "output_dir": str(output_dir),
        "recordings_considered": len(sph_paths),
        "utterances": saved,
        "skipped_empty": skipped_empty,
        "max_recordings": args.max_recordings,
        "max_utterances": args.max_utterances,
        "recording_stem": args.recording_stem,
        "recordings": recordings,
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.summary_out)


if __name__ == "__main__":
    main()
