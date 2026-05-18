#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


def latest_checkpoint(checkpoint_dir):
    paths = sorted(Path(checkpoint_dir).glob("step_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    if not paths:
        raise SystemExit(f"no checkpoints found in {checkpoint_dir}")
    return paths[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--eval-config-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--break-eval", action="store_true")
    parser.add_argument("--include-per-recording", action="store_true")
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    manifest = json.loads(Path(args.run_manifest).read_text(encoding="utf-8"))
    Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
    models = []
    for run in manifest["runs"]:
        ckpt = latest_checkpoint(run["checkpoint_dir"])
        models.append(
            {
                "name": f"rob91_{run['label']}_frozen_ctc_probe",
                "label": run["label"],
                "base_label": run.get("base_label", run["label"]),
                "learning_rate": run.get("learning_rate"),
                "source_ssl_checkpoint": run["local_checkpoint"],
                "path": str(ckpt),
                "seq_len": args.seq_len,
                "overlap_ratio": 0.0,
                "repeat": 1,
            }
        )

    config = {
        "models": models,
        "datasets": [{"name": "tedlium", "splits": ["test"]}],
        "args": {
            "model_class": "SCConformerXL",
            "evaluation_mode": "averaged_moving_window",
            "save_dataframe_path": args.csv_out,
            "verbose": False,
            "disable_flash_attention": False,
            "break_eval": args.break_eval,
            "include_per_recording_evaluations": args.include_per_recording,
        },
    }
    OmegaConf.save(config=OmegaConf.create(config), f=args.eval_config_out)
    print(args.eval_config_out)


if __name__ == "__main__":
    main()
