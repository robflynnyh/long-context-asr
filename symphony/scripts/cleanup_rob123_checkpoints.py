#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path


CHECKPOINT_RE = re.compile(r"step_(\d+)\.pt$")
ALLOWED_PREFIX = "/mnt/parscratch/users/acp21rjf/spotify/"


def parse_checkpoint(path: Path):
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--keep-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not str(checkpoint_dir).startswith(ALLOWED_PREFIX):
        raise ValueError(f"refusing to clean checkpoint dir outside {ALLOWED_PREFIX}: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir does not exist: {checkpoint_dir}")

    checkpoints = []
    for path in checkpoint_dir.iterdir():
        step = parse_checkpoint(path)
        if step is not None:
            checkpoints.append((step, path))
    if not checkpoints:
        raise FileNotFoundError(f"no step_*.pt checkpoints found in {checkpoint_dir}")

    checkpoints.sort(key=lambda item: item[0])
    keep_path = Path(args.keep_checkpoint).resolve() if args.keep_checkpoint else checkpoints[-1][1]
    if keep_path.parent != checkpoint_dir:
        raise ValueError(f"keep checkpoint is not in checkpoint dir: {keep_path}")
    keep_step = parse_checkpoint(keep_path)
    if keep_step is None or not keep_path.exists():
        raise FileNotFoundError(f"keep checkpoint is not a valid existing step_*.pt file: {keep_path}")

    removals = [path for _, path in checkpoints if path != keep_path]
    bytes_to_remove = sum(path.stat().st_size for path in removals)
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"keep_checkpoint={keep_path}")
    print(f"total_step_checkpoints={len(checkpoints)}")
    print(f"remove_count={len(removals)}")
    print(f"remove_bytes={bytes_to_remove}")
    print(f"dry_run={args.dry_run}")
    for path in removals[:10]:
        print(f"remove_candidate={path}")
    if len(removals) > 10:
        print(f"remove_candidate_more={len(removals) - 10}")

    if args.dry_run:
        return

    for path in removals:
        os.remove(path)
    print("cleanup_complete=1")


if __name__ == "__main__":
    main()
