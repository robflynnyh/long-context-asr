#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


ROB119_WER = 0.9961
ROB119_DEL = 0.9244


def read_last_jsonl(path):
    if not path or not Path(path).exists():
        return None
    last = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def percent(value):
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def classify(rows):
    if not rows:
        return "broader setup unresolved: no aggregate evaluation rows were produced"
    best = min(rows, key=lambda item: item["metrics"]["wer"])
    metrics = best["metrics"]
    wer = float(metrics["wer"])
    del_rate = float(metrics.get("del_rate", 1.0))
    hyp_words = int(metrics.get("hyp_words", 0))
    ref_words = int(metrics.get("ref_words", 0))
    length_ratio = hyp_words / ref_words if ref_words else 0.0
    if wer < 0.95 and del_rate < 0.80 and length_ratio > 0.20:
        return "small-adaptation success: top-layer updates escaped the ROB-119 deletion collapse"
    if abs(wer - ROB119_WER) <= 0.01 and del_rate >= 0.85:
        return "fully frozen representation failure still likely: top-layer adaptation did not materially change deletion collapse"
    return "broader SSL/checkpoint/data/eval failure remains plausible: adaptation changed the metrics but did not produce a usable ASR probe"


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
        row = subset.tail(1).iloc[0].to_dict()
        diagnostics = read_last_jsonl(run.get("diagnostics_path"))
        rows.append({"run": run, "metrics": row, "diagnostics": diagnostics})

    lines = [
        "# ROB-126 ROB-100 Top-Layer Unfrozen Probe",
        "",
        "Evidence type: TEDLIUM transfer probe, not direct LibriSpeech paper-comparison evidence.",
        "",
        (
            "Probe setup: ROB-100 SSL backbone with trainable weighted sum over 6 encoder "
            f"hidden states, 2-layer BiLSTM CTC head, and top {manifest.get('unfreeze_top_n_layers')} "
            f"encoder layer(s) unfrozen: {manifest.get('trainable_encoder_layers', [])}."
        ),
        (
            "Training data: TEDLIUM train split as "
            f"`{manifest.get('train_data_format', 'unknown')}` from "
            f"`{manifest.get('train_data_path', 'unknown')}`."
        ),
        (
            f"Optimization: {manifest.get('max_epochs')} epoch(s), scheduler "
            f"`{manifest.get('scheduler', 'constant')}`, warmup steps "
            f"`{manifest.get('warmup_steps', 0)}`, encoder LR scale "
            f"`{manifest.get('encoder_lr_scale')}`."
        ),
        "Checkpoint retention: each trained probe directory is pruned to the latest `step_*.pt` before evaluation.",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| SSL checkpoint | Trainable encoder layers | Epochs | LR | Encoder LR scale | WER | CER | Ins | Del | Sub | Hyp words | Ref words | Final loss | Final blank p | Trained CTC checkpoint |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for item in rows:
            run = item["run"]
            metrics = item["metrics"]
            diagnostics = item["diagnostics"] or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        run["label"],
                        ",".join(map(str, run.get("trainable_encoder_layers", []))),
                        str(run.get("max_epochs", "")),
                        f"{run.get('learning_rate', 0.0):.0e}",
                        str(run.get("encoder_lr_scale", "")),
                        percent(metrics.get("wer")),
                        percent(metrics.get("cer")),
                        percent(metrics.get("ins_rate")),
                        percent(metrics.get("del_rate")),
                        percent(metrics.get("sub_rate")),
                        str(int(metrics.get("hyp_words", 0))),
                        str(int(metrics.get("ref_words", 0))),
                        "" if diagnostics.get("loss") is None else f"{diagnostics['loss']:.4f}",
                        "" if diagnostics.get("blank_p") is None else percent(diagnostics["blank_p"]),
                        f"`{metrics.get('checkpoint')}`",
                    ]
                )
                + " |"
            )
            if run.get("backup_reason"):
                lines.append(f"Backup checkpoint reason for `{run['label']}`: {run['backup_reason']}")
        if manifest.get("mode") == "smoke":
            lines.extend(["", "Comparison to ROB-119: not made from this smoke run; smoke metrics only validate the real path."])
        else:
            best = min(rows, key=lambda item: item["metrics"]["wer"])
            best_metrics = best["metrics"]
            delta_wer = float(best_metrics["wer"]) - ROB119_WER
            delta_del = float(best_metrics.get("del_rate", 0.0)) - ROB119_DEL
            lines.extend(
                [
                    "",
                    (
                        "Comparison to ROB-119: best ROB-126 row is "
                        f"{delta_wer * 100:+.2f} WER points and {delta_del * 100:+.2f} deletion-rate "
                        "points versus ROB-119 frozen weighted BiLSTM "
                        "(99.61% WER, 92.44% deletions)."
                    ),
                    f"Interpretation: {classify(rows)}.",
                ]
            )
    else:
        lines.append("No aggregate TEDLIUM test rows were found.")

    lines.extend(
        [
            "",
            f"Run manifest: `{args.run_manifest}`",
            f"Result CSV: `{args.csv}`",
        ]
    )
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_out)


if __name__ == "__main__":
    main()
