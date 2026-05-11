#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch


CONFIG = "eval_configs_for_thesis/rob69_18l_long_context_finetune.yaml"


def load_checkpoint_metadata(path: str) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        raise RuntimeError(f"Checkpoint has no config payload: {path}")
    model = config.get("model", {})
    if int(model.get("n_layers", 0)) != 18:
        raise RuntimeError(f"Expected 18 layers in {path}, found {model.get('n_layers')}")
    if int(model.get("d_model", 0)) != 1024:
        raise RuntimeError(f"Expected d_model=1024 in {path}, found {model.get('d_model')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    eval_dir = repo_root / "eval"
    os.chdir(eval_dir)
    sys.path.insert(0, str(eval_dir))
    sys.path.insert(0, str(repo_root))

    from eval_manager import checks
    from lcasr.utils.omegaconf import OmegaConf
    import run as run_eval

    config = OmegaConf.load(args.config)
    checks(config, run_eval.datasets_functions)

    datasets_checked = []
    for dataset_config in config.datasets:
        for split in dataset_config.splits:
            rows = run_eval.datasets_functions[dataset_config.name](split)
            if not rows:
                raise RuntimeError(f"{dataset_config.name}/{split} returned no rows")
            first = rows[0]
            for key in ("id", "process_fn"):
                if key not in first:
                    raise RuntimeError(f"{dataset_config.name}/{split} first row missing {key}")
            datasets_checked.append(f"{dataset_config.name}/{split}:{len(rows)}")

    baseline = next(model.path for model in config.models if model.condition == "baseline")
    finetuned = next(model.path for model in config.models if model.condition == "long_only_finetuned")
    load_checkpoint_metadata(baseline)
    load_checkpoint_metadata(finetuned)

    expected = len(config.models) * sum(len(dataset.splits) for dataset in config.datasets)
    print(f"Config OK: {len(config.models)} models, {expected} evaluations")
    print("Datasets OK:", ", ".join(datasets_checked))
    print("Checkpoint metadata OK:", baseline, finetuned)


if __name__ == "__main__":
    main()
