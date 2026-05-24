#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


ROB129_BASELINES = {
    "random_linear": {
        "wer": 0.6717,
        "cer": 0.4875,
        "del_rate": 0.2794,
        "hyp_words": 20855,
        "ref_words": 28215,
        "blank_p": 0.9002,
    },
    "random_bilstm": {
        "wer": 0.3814,
        "cer": 0.2242,
        "del_rate": 0.0984,
        "hyp_words": 26375,
        "ref_words": 28215,
        "blank_p": 0.8263,
    },
}


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
        return "no aggregate evaluation rows were produced"
    bilstm_rows = [item for item in rows if item["run"].get("base_label") == "random_bilstm"]
    target_rows = [item for item in bilstm_rows if item["run"].get("checkpoint_label") == "100pct"] or bilstm_rows
    best = min(target_rows or rows, key=lambda item: item["metrics"]["wer"])
    wer = float(best["metrics"]["wer"])
    del_rate = float(best["metrics"].get("del_rate", 1.0))
    baseline = ROB129_BASELINES["random_bilstm"]
    if wer <= baseline["wer"] + 0.05 and del_rate <= baseline["del_rate"] + 0.05:
        return "the old ROB-91 deletion collapse was mostly explained by the stale TEDLIUM probe path; this weakens masking as the remaining gap"
    if wer >= baseline["wer"] + 0.20 or del_rate >= baseline["del_rate"] + 0.20:
        return "the old low-mask checkpoint remains materially worse than the corrected paper-mask baseline, supporting masking or old-recipe quality as a remaining gap"
    return "the old low-mask checkpoint improves under corrected labels but remains close enough to the paper-mask baseline that the masking conclusion is mixed"


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
        rows.append(
            {
                "run": run,
                "metrics": subset.tail(1).iloc[0].to_dict(),
                "diagnostics": read_last_jsonl(run.get("diagnostics_path")),
            }
        )

    cache_contract = json.loads(Path(manifest["cache_contract"]).read_text(encoding="utf-8"))
    lines = [
        "# ROB-130 Old Low-Mask Corrected-Utterance Probe",
        "",
        "Evidence type: TEDLIUM transfer probe, not direct LibriSpeech paper-comparison evidence.",
        "",
        (
            "Probe setup: fully frozen ROB-70 low-mask SSL backbone, trainable weighted sum over "
            "six exposed encoder hidden states, and trainable CTC probe head(s)."
        ),
        "Hidden-state exposure: six post-layer SCConformerXL states from layers 0-5; the weighted sum includes the final encoder layer state and excludes the pre-layer input.",
        (
            "Training data: corrected TEDLIUM train split as `utterance_folder` from "
            f"`{manifest.get('train_data_path')}`."
        ),
        (
            "Cache contract: sentinel "
            f"`{cache_contract.get('sentinel')}` with `{cache_contract.get('file_count')}` utterance files, "
            f"cleaning `{cache_contract.get('sentinel_cleaning')}`."
        ),
        (
            f"Optimization: {manifest.get('max_epochs')} epoch(s), batch size "
            f"{manifest.get('batch_size')}, scheduler `{manifest.get('scheduler')}`, warmup "
            f"`{manifest.get('warmup_steps')}`."
        ),
        "Checkpoint retention: each trained probe directory is pruned to the latest `step_*.pt` before evaluation.",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| Probe | Source ckpt | Head | Epochs | LR | WER | CER | Ins | Del | Sub | Hyp words | Ref words | Final loss | Final blank p | Trained CTC checkpoint |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
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
                        run.get("checkpoint_file", ""),
                        run.get("probe_head", ""),
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
            lines.extend(["", "Comparison to ROB-129: not made from this smoke run; smoke metrics only validate the real path."])
        else:
            lines.extend(
                [
                    "",
                    "Direct comparison against ROB-129 corrected paper-mask baseline:",
                    "",
                    "| Head | ROB-130 WER | ROB-129 WER | Delta WER | ROB-130 CER | ROB-129 CER | ROB-130 Del | ROB-129 Del | Hyp/ref words |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for item in rows:
                run = item["run"]
                if run.get("checkpoint_label") != "100pct":
                    continue
                baseline = ROB129_BASELINES.get(run.get("base_label"))
                if baseline is None:
                    continue
                metrics = item["metrics"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            run.get("base_label", run["label"]),
                            percent(metrics.get("wer")),
                            percent(baseline["wer"]),
                            f"{(float(metrics.get('wer')) - baseline['wer']) * 100:+.2f} pts",
                            percent(metrics.get("cer")),
                            percent(baseline["cer"]),
                            percent(metrics.get("del_rate")),
                            percent(baseline["del_rate"]),
                            f"{int(metrics.get('hyp_words', 0))}/{int(metrics.get('ref_words', 0))}",
                        ]
                    )
                    + " |"
                )
            lines.extend(
                [
                    f"Interpretation: {classify(rows)}.",
                ]
            )
    else:
        lines.append("No aggregate TEDLIUM test rows were found.")

    lines.extend(
        [
            "",
            "Interpretation boundary: this is corrected-label TEDLIUM transfer evidence for the fully frozen ROB-70 low-mask representation. It should not be treated as a direct LibriSpeech or paper-comparison number.",
            "",
            f"Run manifest: `{args.run_manifest}`",
            f"Result CSV: `{args.csv}`",
        ]
    )
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_out)


if __name__ == "__main__":
    main()
