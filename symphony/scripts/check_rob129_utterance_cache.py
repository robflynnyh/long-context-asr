#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lcasr
from lcasr.utils.dataloading import Utterance_Dataloader


REQUIRED_KEYS = {"id", "audio", "txt", "txt_lengths", "audio_lengths", "text"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance-dir", required=True)
    parser.add_argument("--sentinel", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--loader-batch-size", type=int, default=2)
    args = parser.parse_args()

    utterance_dir = Path(args.utterance_dir)
    sentinel_path = Path(args.sentinel)
    if not utterance_dir.is_dir():
        raise SystemExit(f"missing utterance directory: {utterance_dir}")
    if not sentinel_path.is_file():
        raise SystemExit(f"missing cache sentinel: {sentinel_path}")

    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
    if sentinel.get("cleaning") != "clean_stm_target_v2":
        raise SystemExit(f"unexpected sentinel cleaning value: {sentinel.get('cleaning')}")

    files = sorted(utterance_dir.glob("*.pt"))
    if not files:
        raise SystemExit(f"no utterance .pt files found in {utterance_dir}")
    expected = int(sentinel.get("utterances", len(files)))
    if expected != len(files):
        raise SystemExit(f"sentinel utterance count {expected} != file count {len(files)}")

    samples = []
    for path in files[: args.sample_count]:
        item = torch.load(path, map_location="cpu", weights_only=False)
        missing = REQUIRED_KEYS - set(item)
        if missing:
            raise SystemExit(f"{path} missing required key(s): {sorted(missing)}")
        audio = item["audio"]
        txt = item["txt"]
        audio_lengths = item["audio_lengths"]
        txt_lengths = item["txt_lengths"]
        if audio.ndim != 3 or audio.shape[0] != 1:
            raise SystemExit(f"{path} audio must have shape [1, mel, frames], got {tuple(audio.shape)}")
        if txt.ndim != 2 or txt.shape[0] != 1:
            raise SystemExit(f"{path} txt must have shape [1, tokens], got {tuple(txt.shape)}")
        if int(audio_lengths[0]) > audio.shape[-1]:
            raise SystemExit(f"{path} audio_lengths exceeds padded audio frames")
        if int(txt_lengths[0]) > txt.shape[-1]:
            raise SystemExit(f"{path} txt_lengths exceeds padded txt tokens")
        samples.append(
            {
                "file": str(path),
                "id": item["id"],
                "text": item["text"],
                "audio_frames": int(audio_lengths[0]),
                "tokens": int(txt_lengths[0]),
            }
        )

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    dataloader = Utterance_Dataloader(
        utterance_folder=str(utterance_dir),
        tokenizer=tokenizer,
        batch_size=args.loader_batch_size,
        num_workers=0,
        prefetch=1,
        pin_memory=False,
        shuffle=False,
        max_records=max(args.loader_batch_size, 1),
    )
    batch = next(iter(dataloader))
    batch_keys = sorted(batch.keys())
    expected_batch_keys = ["audio", "audio_lengths", "ids", "text", "text_lengths"]
    if batch_keys != expected_batch_keys:
        raise SystemExit(f"unexpected Utterance_Dataloader batch keys: {batch_keys}")

    summary = {
        "issue": "ROB-129",
        "utterance_dir": str(utterance_dir),
        "sentinel": str(sentinel_path),
        "sentinel_cleaning": sentinel.get("cleaning"),
        "sentinel_utterances": expected,
        "file_count": len(files),
        "loader_contract": {
            "sample_required_keys": sorted(REQUIRED_KEYS),
            "batch_keys": batch_keys,
            "batch_size": int(batch["audio"].shape[0]),
            "audio_shape": list(batch["audio"].shape),
            "text_shape": list(batch["text"].shape),
        },
        "samples": samples,
    }
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.summary_out)


if __name__ == "__main__":
    main()
