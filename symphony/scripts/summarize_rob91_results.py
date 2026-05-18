#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def classify(rows):
    if len(rows) < 3:
        return "inconclusive: fewer than three checkpoint probes completed"
    ordered = sorted(rows, key=lambda row: {"25pct": 0, "50pct": 1, "100pct": 2}[row["label"]])
    wers = [row["wer"] for row in ordered]
    if wers[-1] < wers[0] and wers[1] <= max(wers[0], wers[-1]):
        return "promising: later SSL checkpoints improve over the 25% checkpoint"
    if wers[-1] >= wers[0]:
        return "negative: the final SSL checkpoint does not improve over the 25% checkpoint"
    return "inconclusive: WER changes are not monotonic enough to read confidently"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.run_manifest).read_text(encoding="utf-8"))
    df = pd.read_csv(args.csv)
    rows = []
    for run in manifest["runs"]:
        subset = df[(df["recording"] == "all") & (df["label"] == run["label"])]
        if subset.empty:
            continue
        row = subset.tail(1).iloc[0]
        rows.append(
            {
                "label": run["label"],
                "wer": float(row["wer"]),
                "checkpoint": row["checkpoint"],
                "source_ssl_checkpoint": run["local_checkpoint"],
            }
        )

    lines = ["# ROB-91 Frozen BEST-RQ CTC Probe", ""]
    if rows:
        lines.extend(["| SSL checkpoint | TEDLIUM test WER | Trained CTC checkpoint |", "| --- | ---: | --- |"])
        for row in sorted(rows, key=lambda row: {"25pct": 0, "50pct": 1, "100pct": 2}[row["label"]]):
            lines.append(f"| {row['label']} | {row['wer'] * 100:.2f}% | `{row['checkpoint']}` |")
        lines.extend(["", f"Interpretation: {classify(rows)}."])
    else:
        lines.append("No aggregate TEDLIUM test WER rows were found.")

    lines.extend(["", f"Result CSV: `{args.csv}`"])
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_out)


if __name__ == "__main__":
    main()
