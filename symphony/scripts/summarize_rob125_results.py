#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


ROB119_FINAL_WER = 0.9961


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
        "# ROB-125 Known-Good Supervised Encoder Probe",
        "",
        "Evidence type: TEDLIUM transfer sanity probe. This checks whether the ROB-119-style probe/eval path can emit words when the frozen encoder is already supervised-ASR useful.",
        "",
        "Probe setup: frozen ROB-81 supervised Floras finetune encoder, trainable weighted sum over encoder hidden states, 2-layer BiLSTM CTC head with hidden size 1024 and dropout 0.2.",
        "Known-good evidence: ROB-81 TEDLIUM eval reports 8.258% WER over 28,215 words for the source checkpoint.",
        f"Optimization: {manifest.get('max_epochs')} epochs, scheduler `{manifest.get('scheduler', 'constant')}`, warmup steps `{manifest.get('warmup_steps', 0)}`.",
        "Checkpoint retention: each trained probe directory is pruned to the latest `step_*.pt` before evaluation.",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| Source checkpoint | Epochs | LR | WER | CER | Ins | Del | Sub | Hyp words | Ref words | Final loss | Final blank p | Trained CTC checkpoint |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                        str(run.get("max_epochs", "")),
                        f"{run.get('learning_rate', 0.0):.0e}",
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
        if manifest.get("mode") == "smoke":
            lines.extend(["", "Harness read: smoke metrics only validate the real path; use a longer run for a final sanity conclusion."])
        else:
            best = min(rows, key=lambda item: item["metrics"]["wer"])
            best_wer = float(best["metrics"]["wer"])
            delta = best_wer - ROB119_FINAL_WER
            if best_wer < ROB119_FINAL_WER:
                comparison = f"emitted more useful words than ROB-119; WER is {-delta * 100:.2f} points below the ROB-119 frozen SSL probe"
            elif best_wer > ROB119_FINAL_WER:
                comparison = f"did not beat the ROB-119 frozen SSL probe; WER is {delta * 100:.2f} points higher"
            else:
                comparison = "matched the ROB-119 frozen SSL probe"
            harness = "harness can produce nonblank words with a known-good frozen supervised encoder" if best["metrics"].get("hyp_words", 0) > 5000 else "harness remains deletion-heavy even with a known-good frozen supervised encoder"
            lines.extend(["", f"Comparison to ROB-119: {comparison} (ROB-119 reference WER 99.61%).", f"Harness conclusion: {harness}."])
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
