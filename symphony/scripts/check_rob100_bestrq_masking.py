#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lcasr.models.BestRQ import BestRQ


class DummyAcousticModel(nn.Module):
    def __init__(self, feat_in: int, d_model: int, downsampling_factor: int):
        super().__init__()
        self.feat_in = feat_in
        self.d_model = d_model
        self.downsampling_factor = downsampling_factor
        self.proj = nn.Linear(feat_in * downsampling_factor, d_model)

    def forward(self, audio_signal, length=None, skip_vocab_projection=True):
        batch, feat, time = audio_signal.shape
        stacked = audio_signal.transpose(1, 2).reshape(
            batch,
            time // self.downsampling_factor,
            feat * self.downsampling_factor,
        )
        return {"hidden_states": self.proj(stacked)}


@dataclass
class MaskCase:
    name: str
    kwargs: dict
    expected_min: float
    expected_max: float


def run_case(case: MaskCase, repeats: int, stacked_frames: int, seed: int):
    torch.manual_seed(seed)
    downsampling_factor = 8
    model = DummyAcousticModel(
        feat_in=80,
        d_model=32,
        downsampling_factor=downsampling_factor,
    )
    best_rq = BestRQ(
        model=model,
        downsampling_factor=downsampling_factor,
        codebook_size=64,
        codebook_dim=8,
        **case.kwargs,
    )
    audio = torch.randn(1, 80, stacked_frames * downsampling_factor)
    lengths = torch.tensor([audio.shape[-1]], dtype=torch.long)
    ratios = []
    masked = []
    valid = []
    skipped = 0
    for _ in range(repeats):
        out = best_rq(audio, length=lengths)
        ratios.append(float(out["actual_mask_ratio"]))
        masked.append(int(out["masked_stacked_frames"]))
        valid.append(int(out["valid_stacked_frames"]))
        skipped += int(out.get("skipped_empty_mask", 0))

    mean_ratio = sum(ratios) / len(ratios)
    result = {
        "case": case.name,
        "repeats": repeats,
        "stacked_frames": stacked_frames,
        "valid_stacked_frames": valid[0],
        "min_masked_stacked_frames": min(masked),
        "max_masked_stacked_frames": max(masked),
        "mean_actual_mask_ratio": mean_ratio,
        "min_actual_mask_ratio": min(ratios),
        "max_actual_mask_ratio": max(ratios),
        "skipped_empty_mask_count": skipped,
        "expected_min": case.expected_min,
        "expected_max": case.expected_max,
        "passed": case.expected_min <= mean_ratio <= case.expected_max and skipped == 0,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--stacked-frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cases = [
        MaskCase(
            name="legacy_lowmask",
            kwargs={"mask_mode": "legacy_groups", "mask_percentage": 0.1, "frames_to_mask": 5},
            expected_min=0.07,
            expected_max=0.13,
        ),
        MaskCase(
            name="paper_mask_p012_l4",
            kwargs={"mask_mode": "speechbrain", "mask_prob": 0.12, "mask_length": 4},
            expected_min=0.45,
            expected_max=0.50,
        ),
        MaskCase(
            name="open_mask_p015_l4",
            kwargs={"mask_mode": "speechbrain", "mask_prob": 0.15, "mask_length": 4},
            expected_min=0.57,
            expected_max=0.62,
        ),
    ]

    results = [
        run_case(
            case=case,
            repeats=args.repeats,
            stacked_frames=args.stacked_frames,
            seed=args.seed + index,
        )
        for index, case in enumerate(cases)
    ]
    print(json.dumps(results, indent=2))
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
