#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


LABEL_ORDER = {"25pct": 0, "50pct": 1, "100pct": 2}


def sort_key(row):
    return (LABEL_ORDER.get(row.get("base_label", row["label"]), 99), row.get("learning_rate") or 0.0, row["label"])


def classify(rows):
    comparison_rows = [row for row in rows if row.get("base_label") in LABEL_ORDER]
    learning_rates = {row.get("learning_rate") for row in comparison_rows}
    base_labels = {row.get("base_label") for row in comparison_rows}
    if len(comparison_rows) < 3 or base_labels != set(LABEL_ORDER):
        return "inconclusive: fewer than three checkpoint probes completed"
    if len(learning_rates) != 1:
        return "inconclusive: rows span multiple learning rates, so checkpoint progression is not directly comparable"
    ordered = sorted(comparison_rows, key=lambda row: LABEL_ORDER[row["base_label"]])
    wers = [row["wer"] for row in ordered]
    if wers[-1] < wers[0] and wers[1] <= max(wers[0], wers[-1]):
        if wers[-1] >= 0.9:
            return "inconclusive: later SSL checkpoints improve slightly, but all probes remain near 100% WER"
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
                "base_label": run.get("base_label", run["label"]),
                "learning_rate": run.get("learning_rate"),
                "probe_head": run.get("probe_head", "linear"),
                "wer": float(row["wer"]),
                "checkpoint": row["checkpoint"],
                "source_ssl_checkpoint": run["local_checkpoint"],
            }
        )

    lines = ["# ROB-91 Frozen BEST-RQ CTC Probe", ""]
    if rows:
        lines.extend(["| SSL checkpoint | Probe head | LR | TEDLIUM test WER | Trained CTC checkpoint |", "| --- | --- | ---: | ---: | --- |"])
        for row in sorted(rows, key=sort_key):
            lr = "" if row["learning_rate"] is None else f"{row['learning_rate']:.0e}"
            lines.append(f"| {row['label']} | {row['probe_head']} | {lr} | {row['wer'] * 100:.2f}% | `{row['checkpoint']}` |")
        lines.extend(["", f"Interpretation: {classify(rows)}."])
    else:
        lines.append("No aggregate TEDLIUM test WER rows were found.")

    lines.extend(["", f"Result CSV: `{args.csv}`"])
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_out)


if __name__ == "__main__":
    main()
