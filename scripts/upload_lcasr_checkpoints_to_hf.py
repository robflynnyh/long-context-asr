#!/usr/bin/env python3
"""Upload LCASR checkpoint groups to Hugging Face.

This script preserves the checkpoint directory layout used by the existing
`rjflynn2/lcasr-6L-768D-6H-RB-1p5M` model repo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


USER = "rjflynn2"


@dataclass(frozen=True)
class UploadItem:
    local_path: Path
    path_in_repo: str


@dataclass(frozen=True)
class UploadGroup:
    key: str
    repo_id: str
    title: str
    description: str
    source_config: str
    items: tuple[UploadItem, ...]


def seq_scheduler_items(root: str, seq_lens: Iterable[int], repeats: Iterable[int], checkpoint: str) -> tuple[UploadItem, ...]:
    root_path = Path(root)
    items: list[UploadItem] = []
    for seq_len in seq_lens:
        for repeat in repeats:
            rel = f"n_seq_sched_{seq_len}_rp_{repeat}/{checkpoint}"
            items.append(UploadItem(root_path / rel, rel))
    return tuple(items)


def window_items(root: str, windows: Iterable[int], repeats: Iterable[int], checkpoint: str) -> tuple[UploadItem, ...]:
    root_path = Path(root)
    items: list[UploadItem] = []
    for window in windows:
        for repeat in repeats:
            rel = f"rb_window_size_{window}_rp_{repeat}/{checkpoint}"
            items.append(UploadItem(root_path / rel, rel))
    return tuple(items)


def spotify_ft_items(root: str, windows: Iterable[int], repeats: Iterable[int], lr_name: str, checkpoint: str) -> tuple[UploadItem, ...]:
    root_path = Path(root)
    items: list[UploadItem] = []
    for window in windows:
        for repeat in repeats:
            rel = f"spotify-L-FT-w{window}-{lr_name}-rp-{repeat}/{checkpoint}"
            items.append(UploadItem(root_path / rel, rel))
    return tuple(items)


GROUPS: dict[str, UploadGroup] = {
    "18l": UploadGroup(
        key="18l",
        repo_id=f"{USER}/lcasr-18L-1024D-8H-RB-1p5M",
        title="LCASR 18L 1024D 8H RB 1.5M",
        description="SCConformerXL 18-layer, 1024-dimensional, 8-head rotary-base 1.5M Spotify long-context checkpoints.",
        source_config="exp/configs/paper_templates/exp_set_seq_rotary_base_18l.yaml",
        items=seq_scheduler_items(
            "/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_rb_18l_1024D",
            [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 360000],
            [1, 2, 3],
            "step_105360.pt",
        ),
    ),
    "6epoch": UploadGroup(
        key="6epoch",
        repo_id=f"{USER}/lcasr-6L-768D-6H-RB-1p5M-6epoch-windowed",
        title="LCASR 6L 768D 6H RB 1.5M 6 Epoch Windowed",
        description="SCConformerXL 6-layer, 768-dimensional, 6-head rotary-base 1.5M checkpoints trained for 6 epochs with windowed attention variants.",
        source_config="exp/configs/paper_templates/exp_set_seq_window_sizes_6epochs.yaml",
        items=window_items(
            "/mnt/parscratch/users/acp21rjf/spotify/checkpoints_seq_scheduler_window_sizes_6epochs",
            [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 360000],
            [1, 2, 3],
            "step_632160.pt",
        ),
    ),
    "long-ft-5e-5": UploadGroup(
        key="long-ft-5e-5",
        repo_id=f"{USER}/lcasr-6L-768D-6H-RB-1p5M-long-context-FT-5e-5",
        title="LCASR 6L 768D 6H RB 1.5M Long-Context FT 5e-5",
        description="SCConformerXL long-context-only fine-tuned checkpoints at learning rate 5e-5.",
        source_config="eval/eval_configs_for_thesis/eval_config_pre_windowed_spotify_FT_long_only.yaml",
        items=spotify_ft_items(
            "/mnt/parscratch/users/acp21rjf/spotify/long_only/FT_3epoch",
            [128, 512, 2048, 8192, 22500],
            [1, 2, 3],
            "5e5",
            "step_23763.pt",
        ),
    ),
    "18l-long-ft-5e-5": UploadGroup(
        key="18l-long-ft-5e-5",
        repo_id=f"{USER}/lcasr-18L-1024D-8H-RB-1p5M-long-context-FT-5e-5",
        title="LCASR 18L 1024D 8H RB 1.5M Long-Context FT 5e-5",
        description="SCConformerXL 18-layer long-context-only fine-tuned checkpoints at learning rate 5e-5.",
        source_config="exp/configs/paper_templates/exp_set_spotify_L_FT.yaml",
        items=spotify_ft_items(
            "/mnt/parscratch/users/acp21rjf/spotify/long_only/FT_3epoch_18L",
            [128, 512, 2048, 8192, 22500],
            [1, 2, 3],
            "5e5",
            "step_23763.pt",
        ),
    ),
}


def readme_for(group: UploadGroup) -> str:
    return f"""---
license: apache-2.0
---

# {group.title}

{group.description}

Source config in the LCASR repository: `{group.source_config}`.

The checkpoint directory layout mirrors the local training output layout.
"""


def existing_files(api: HfApi, repo_id: str) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception:
        return set()


def upload_group(api: HfApi, group: UploadGroup, dry_run: bool, private: bool) -> None:
    print(f"\n== {group.key}: {group.repo_id} ==")
    missing = [item for item in group.items if not item.local_path.is_file()]
    if missing:
        for item in missing:
            print(f"missing: {item.local_path}")
        raise FileNotFoundError(f"{len(missing)} checkpoint files are missing for {group.key}")

    total_size = sum(item.local_path.stat().st_size for item in group.items)
    print(f"files: {len(group.items)}")
    print(f"bytes: {total_size}")

    if dry_run:
        for item in group.items:
            print(f"dry-run: {item.local_path} -> {item.path_in_repo}")
        return

    api.create_repo(repo_id=group.repo_id, repo_type="model", private=private, exist_ok=True)
    present = existing_files(api, group.repo_id)

    attrs = "*.pt filter=lfs diff=lfs merge=lfs -text\n"
    if ".gitattributes" not in present:
        api.upload_file(
            repo_id=group.repo_id,
            repo_type="model",
            path_or_fileobj=attrs.encode("utf-8"),
            path_in_repo=".gitattributes",
            commit_message="Add LFS attributes",
        )

    api.upload_file(
        repo_id=group.repo_id,
        repo_type="model",
        path_or_fileobj=readme_for(group).encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Update README",
    )

    present = existing_files(api, group.repo_id)
    for idx, item in enumerate(group.items, start=1):
        if item.path_in_repo in present:
            print(f"[{idx}/{len(group.items)}] skip existing {item.path_in_repo}")
            continue
        print(f"[{idx}/{len(group.items)}] upload {item.local_path} -> {item.path_in_repo}")
        api.upload_file(
            repo_id=group.repo_id,
            repo_type="model",
            path_or_fileobj=str(item.local_path),
            path_in_repo=item.path_in_repo,
            commit_message=f"Upload {item.path_in_repo}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), action="append", help="Group to upload. Defaults to all groups.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--private", action="store_true", help="Create new repos as private.")
    args = parser.parse_args()

    selected = args.group or list(GROUPS)
    api = HfApi()
    for key in selected:
        upload_group(api, GROUPS[key], dry_run=args.dry_run, private=args.private)


if __name__ == "__main__":
    main()
