#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm

from lcasr.utils.audio_tools import processing_chain, total_seconds


def normalize_stm_text(text: str) -> str:
    text = re.sub(r" '([a-z])", r"'\1", text)
    return re.sub(r" +", " ", text).strip()


def read_stm_segments(stm_path: Path):
    segments = []
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
            segments.append({"start": start, "end": end, "text": text})
    return segments


def build_manifest(args):
    split_root = Path(args.tedlium_root) / args.split
    sph_dir = split_root / "sph"
    stm_dir = split_root / "stm"
    if not sph_dir.exists():
        raise SystemExit(f"missing sph directory: {sph_dir}")
    if not stm_dir.exists():
        raise SystemExit(f"missing stm directory: {stm_dir}")

    spec_dir = Path(args.spec_dir)
    transcript_dir = Path(args.transcript_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)

    sph_files = sorted(sph_dir.glob("*.sph"))
    if args.max_records is not None:
        sph_files = sph_files[: args.max_records]

    manifest = {}
    for sph_path in tqdm(sph_files, desc=f"Preparing TEDLIUM {args.split}"):
        stem = sph_path.stem
        stm_path = stm_dir / f"{stem}.stm"
        if not stm_path.exists():
            raise SystemExit(f"missing STM for {sph_path}: {stm_path}")

        spec_path = spec_dir / f"{stem}.spec.pt"
        transcript_path = transcript_dir / f"{stem}.json"

        if args.force or not spec_path.exists():
            spec = processing_chain(str(sph_path))
            torch.save(spec, spec_path)
        else:
            spec = torch.load(spec_path, map_location="cpu")

        segments = read_stm_segments(stm_path)
        if args.force or not transcript_path.exists():
            transcript_path.write_text(
                json.dumps({"word_timestamps": segments}, ensure_ascii=True),
                encoding="utf-8",
            )

        manifest[stem] = {
            "audio": str(spec_path),
            "txt": str(transcript_path),
            "duration": total_seconds(int(spec.shape[-1])),
        }

    with open(args.manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"wrote {len(manifest)} records to {args.manifest_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tedlium-root", default="/store/store4/data/TEDLIUM_release1/legacy")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--transcript-dir", required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--force", action="store_true")
    build_manifest(parser.parse_args())


if __name__ == "__main__":
    main()
