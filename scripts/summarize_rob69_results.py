#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


RESULT_CSV = Path("eval/results/thesis/rob69_18l_long_context_finetune_vs_baseline.csv")
WINDOWS = [128, 512, 2048, 8192, 22500]
DATASETS = ["tedlium", "rev16", "earnings22_full", "this_american_life", "earnings22"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=RESULT_CSV)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.loc[(df["recording"] == "all") & (df["split"] == "test")].copy()
    expected = len(DATASETS) * len(WINDOWS) * 3 * 2
    if len(df) != expected:
        raise SystemExit(f"Expected {expected} aggregate comparison rows, found {len(df)}")

    grouped = (
        df.groupby(["dataset", "condition", "window_size"], as_index=False)
        .agg(wer_mean=("wer", "mean"), wer_std=("wer", "std"), n=("wer", "size"))
        .sort_values(["dataset", "window_size", "condition"])
    )
    pivot = grouped.pivot_table(
        index=["dataset", "window_size"],
        columns="condition",
        values="wer_mean",
        aggfunc="first",
    ).reset_index()
    pivot["absolute_delta"] = pivot["long_only_finetuned"] - pivot["baseline"]
    pivot["relative_delta_pct"] = 100.0 * pivot["absolute_delta"] / pivot["baseline"]

    lines = [
        "# ROB-69 18L long-context finetuning benchmark",
        "",
        "Mean WER across 3 repeats; negative delta means the long-only finetuned checkpoint improved over the matched 18L baseline.",
        "",
        "| dataset | window_size | baseline_wer | finetuned_wer | abs_delta | rel_delta_pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pivot.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.window_size)} | {row.baseline:.6f} | "
            f"{row.long_only_finetuned:.6f} | {row.absolute_delta:.6f} | {row.relative_delta_pct:.2f} |"
        )
    text = "\n".join(lines) + "\n"

    print(text)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(text)


if __name__ == "__main__":
    main()
