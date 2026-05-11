#!/usr/bin/env python3
"""Expand the ROB-78 template into deterministic Slurm-ready configs."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


DEFAULT_TEMPLATE = Path("exp/configs/paper_templates/exp_set_spotify_f_long_only_ft_6epoch.yaml")
DEFAULT_ARTIFACT_ROOT = Path("/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-78")


def get_nested(config: Any, dotted_key: str) -> Any:
    cur = config
    for part in dotted_key.split("."):
        cur = cur[part]
    return cur


def set_nested(config: Any, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = config
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def expand_template(template_path: Path) -> list[Any]:
    template = OmegaConf.load(template_path)
    include_keys = list(template["template_info"]["include_keys"])
    template_keys = list(template["template_info"]["template_keys"])
    count = int(template["template_info"]["create"])
    configs = []
    for idx in range(count):
        config = OmegaConf.create({})
        for key in include_keys:
            config[key] = copy.deepcopy(template[key])
        for dotted_key in template_keys:
            set_nested(config, dotted_key, copy.deepcopy(get_nested(config, dotted_key)[idx]))
        if "wandb" in config and "update_config_with_wandb_id" not in config["wandb"]:
            config["wandb"]["update_config_with_wandb_id"] = True
        configs.append(config)
    return configs


def apply_smoke_overrides(config: Any, artifact_root: Path) -> Any:
    config = copy.deepcopy(config)
    config["wandb"]["use"] = False
    config["wandb"]["name"] = "rob78-spotify-f-long-only-smoke"
    config["wandb"]["id"] = ""
    config["checkpointing"]["dir"] = str(artifact_root / "smoke-checkpoints")
    config["checkpointing"]["save_every_n_steps"] = 1000000
    config["audio_chunking"]["size"] = 512
    config["training"]["batch_size"] = 1
    config["training"]["max_epochs"] = 1
    config["training"]["max_steps"] = 1
    config["training"]["random_seed"] = 12345
    return config


def resolve_words(text_json: dict[str, Any]) -> list[dict[str, Any]]:
    if "word_timestamps" in text_json:
        return text_json["word_timestamps"]
    return text_json["results"][-1]["alternatives"][0]["words"]


def word_times(word: dict[str, Any]) -> tuple[float, float]:
    if "startTime" in word and "endTime" in word:
        return float(word["startTime"][:-1]), float(word["endTime"][:-1])
    return float(word["start"]), float(word["end"])


def build_smoke_manifest(config: Any, artifact_root: Path, frames: int = 512) -> Path:
    import torch

    source_manifest = Path(str(config["data"]["path"]))
    smoke_dir = artifact_root / "smoke-data"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    seconds = frames * 160 / 16000
    pairs = json.loads(source_manifest.read_text(encoding="utf-8"))

    for sample_id, entry in pairs.items():
        txt_path = Path(entry["txt"])
        words = resolve_words(json.loads(txt_path.read_text(encoding="utf-8")))
        if not [word for word in words if word_times(word)[1] <= seconds]:
            continue
        audio = torch.load(entry["audio"], map_location="cpu")
        if audio.shape[-1] < frames:
            continue
        smoke_audio = smoke_dir / "audio_512f.pt"
        smoke_manifest = smoke_dir / "pairs.json"
        torch.save(audio[..., :frames].contiguous(), smoke_audio)
        smoke_pairs = {
            f"{sample_id}_rob78_smoke": {
                **entry,
                "audio": str(smoke_audio),
                "txt": str(txt_path),
                "duration": seconds,
            }
        }
        smoke_manifest.write_text(json.dumps(smoke_pairs, indent=2) + "\n", encoding="utf-8")
        return smoke_manifest
    raise RuntimeError(f"could not build a {frames}-frame smoke manifest from {source_manifest}")


def write_configs(configs: list[Any], output_dir: Path, manifest_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, config in enumerate(configs):
        name = safe_name(str(config["wandb"]["name"]))
        path = output_dir / f"{idx:02d}_{name}.yaml"
        OmegaConf.save(config=config, f=path)
        paths.append(path)
    manifest_path.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8")
    print(f"wrote {len(paths)} configs")
    print(f"manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--mode", choices=["train", "smoke"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configs = expand_template(args.template)
    if args.mode == "smoke":
        # CPU attention in this repo supports only full attention, so use the
        # full-context member of the same ROB-78 config family for the smoke.
        smoke_config = apply_smoke_overrides(configs[-1], args.artifact_root)
        smoke_config["data"]["path"] = str(build_smoke_manifest(smoke_config, args.artifact_root))
        configs = [smoke_config]
        output_dir = args.artifact_root / "smoke-configs"
        manifest = args.artifact_root / "smoke_configs.txt"
    else:
        if args.limit is not None:
            configs = configs[: args.limit]
        output_dir = args.artifact_root / "train-configs"
        manifest = args.artifact_root / "train_configs.txt"
    write_configs(configs, output_dir, manifest)


if __name__ == "__main__":
    main()
