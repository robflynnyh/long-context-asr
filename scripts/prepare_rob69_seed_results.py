#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd


BASELINE_CSV = Path("eval/results/model_sizes/evals_windowed_18l_1024D.csv")
OUTPUT_CSV = Path("eval/results/thesis/rob69_18l_long_context_finetune_vs_baseline.csv")

WINDOW_BY_SEQ_LEN = {
    2048: 128,
    8192: 512,
    32768: 2048,
    131072: 8192,
    360000: 22500,
}

OUTPUT_COLUMNS = [
    "dataset",
    "split",
    "wer",
    "recording",
    "words",
    "ins_rate",
    "del_rate",
    "sub_rate",
    "name",
    "checkpoint",
    "repeat",
    "seq_len",
    "overlap_ratio",
    "model_class",
    "condition",
    "matched_baseline_seq_len",
    "window_size",
    "eval_group",
    "training_config",
]


def build_seed_rows(baseline_csv: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_csv)
    rows = baseline.loc[
        (baseline["recording"] == "all")
        & (baseline["split"] == "test")
        & (baseline["seq_len"].isin(WINDOW_BY_SEQ_LEN))
    ].copy()
    if rows.empty:
        raise SystemExit(f"No matching baseline rows found in {baseline_csv}")

    rows = rows.drop(columns=[col for col in rows.columns if col.startswith("Unnamed")], errors="ignore")
    rows = rows.drop_duplicates(["dataset", "split", "seq_len", "repeat"], keep="last")
    rows["name"] = "rb_18l_1024D_baseline"
    rows["condition"] = "baseline"
    rows["matched_baseline_seq_len"] = rows["seq_len"].astype(int)
    rows["window_size"] = rows["matched_baseline_seq_len"].map(WINDOW_BY_SEQ_LEN)
    rows["eval_group"] = "rob69_18l_long_context_finetune"
    rows["training_config"] = "exp/configs/paper_templates/exp_set_seq_rotary_base_18l.yaml"
    return rows[OUTPUT_COLUMNS].sort_values(
        ["dataset", "split", "matched_baseline_seq_len", "repeat", "condition"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_csv.exists() and not args.force:
        existing = pd.read_csv(args.output_csv)
        required = {"condition", "matched_baseline_seq_len", "window_size", "eval_group"}
        missing = sorted(required - set(existing.columns))
        if missing:
            raise SystemExit(f"{args.output_csv} exists but is missing columns: {missing}")
        print(f"Keeping existing seeded/evaluation CSV: {args.output_csv}")
        return

    rows = build_seed_rows(args.baseline_csv)
    os.makedirs(args.output_csv.parent, exist_ok=True)
    rows.to_csv(args.output_csv, index=False)
    expected = len(WINDOW_BY_SEQ_LEN) * 3 * 5
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} baseline rows, wrote {len(rows)}")
    print(f"Wrote {len(rows)} baseline seed rows to {args.output_csv}")


if __name__ == "__main__":
    main()
