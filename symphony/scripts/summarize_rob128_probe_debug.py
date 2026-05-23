#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def percent(value):
    return f"{float(value) * 100:.2f}%"


def read_last_jsonl(path):
    if not path or not Path(path).exists():
        return None
    last = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last


def eval_only_run(label, eval_result):
    return {
        "label": label,
        "max_epochs": "eval-only",
        "learning_rate": None,
        "eval_only": True,
        "checkpoint": eval_result.get("checkpoint", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--pass-json-out")
    parser.add_argument("--pass-wer-threshold", type=float, default=0.20)
    parser.add_argument("--pass-blank-threshold", type=float, default=0.60)
    args = parser.parse_args()

    manifest = json.loads(Path(args.run_manifest).read_text(encoding="utf-8"))
    eval_dir = Path(args.eval_dir)
    rows = []
    for run in manifest["runs"]:
        eval_path = eval_dir / f"{run['label']}.json"
        if not eval_path.exists():
            rows.append({"run": run, "eval": None, "diagnostics": read_last_jsonl(run.get("diagnostics_path"))})
            continue
        rows.append(
            {
                "run": run,
                "eval": json.loads(eval_path.read_text(encoding="utf-8")),
                "diagnostics": read_last_jsonl(run.get("diagnostics_path")),
            }
        )
    seen_labels = {row["run"]["label"] for row in rows}
    for label in ["source_ctc_eval_only"]:
        if label in seen_labels:
            continue
        eval_path = eval_dir / f"{label}.json"
        if not eval_path.exists():
            continue
        result = json.loads(eval_path.read_text(encoding="utf-8"))
        rows.append({"run": eval_only_run(label, result), "eval": result, "diagnostics": None})

    primary = next((row for row in rows if row["run"]["label"] == "random_bilstm"), rows[0] if rows else None)
    primary_eval = primary["eval"] if primary else None
    passed = bool(
        primary_eval
        and float(primary_eval["wer"]) <= args.pass_wer_threshold
        and float(primary_eval["blank_rate"]) <= args.pass_blank_threshold
        and int(primary_eval["hyp_words"]) > 0
    )

    lines = [
        "# ROB-128 Supervised-Feature Probe Debug",
        "",
        "Evidence type: known-good ROB-81 supervised encoder with TEDLIUM train examples cut on STM utterance boundaries.",
        "",
        f"Stage: `{manifest.get('stage')}`",
        f"Mode: `{manifest.get('mode')}`",
        f"Training data: `{manifest.get('train_data_path')}`",
        f"Utterance summary: `{manifest.get('utterance_summary')}`",
        f"Supervised checkpoint: `{manifest.get('supervised_checkpoint')}`",
        "",
        "| Probe | Epochs | LR | CTC loss/utt | Blank | WER | CER | Hyp words | Ref words | Input frames | Output frames | Target tokens | Checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        run = row["run"]
        result = row["eval"] or {}
        diagnostics = row["diagnostics"] or {}
        input_lengths = result.get("input_lengths", [])
        output_lengths = result.get("output_lengths", [])
        target_lengths = result.get("target_lengths", [])
        lines.append(
            "| "
            + " | ".join(
                [
                    run["label"],
                    str(run.get("max_epochs", "")),
                    "eval-only" if run.get("learning_rate") is None else f"{run.get('learning_rate', 0.0):.0e}",
                    "" if result.get("ctc_loss_per_utterance") is None else f"{result['ctc_loss_per_utterance']:.4f}",
                    "" if result.get("blank_rate") is None else percent(result["blank_rate"]),
                    "" if result.get("wer") is None else percent(result["wer"]),
                    "" if result.get("cer") is None else percent(result["cer"]),
                    str(int(result.get("hyp_words", 0))),
                    str(int(result.get("ref_words", 0))),
                    f"{min(input_lengths, default=0)}-{max(input_lengths, default=0)}",
                    f"{min(output_lengths, default=0)}-{max(output_lengths, default=0)}",
                    f"{min(target_lengths, default=0)}-{max(target_lengths, default=0)}",
                    f"`{result.get('checkpoint', '')}`",
                ]
            )
            + " |"
        )
        if diagnostics:
            lines.append(
                f"Last train diagnostic for `{run['label']}`: loss={diagnostics.get('loss')}, blank_p={diagnostics.get('blank_p')}, step={diagnostics.get('step')}."
            )

    if primary_eval:
        lines.extend(
            [
                "",
                f"One-record overfit result: {'PASS' if passed else 'FAIL'}.",
                (
                    f"Pass rule: random_bilstm WER <= {percent(args.pass_wer_threshold)}, "
                    f"blank rate <= {percent(args.pass_blank_threshold)}, and nonzero hypothesis words."
                ),
                "",
                "Greedy transcript from primary probe:",
                primary_eval.get("greedy_transcript", ""),
                "",
                "Reference transcript:",
                primary_eval.get("reference_transcript", ""),
            ]
        )
    else:
        lines.extend(["", "One-record overfit result: FAIL. No primary eval JSON was produced."])

    lines.extend(
        [
            "",
            "Interpretation boundary: this result tests fresh probe optimization on known-good supervised features. It does not by itself judge ROB-100 SSL representation quality, and it is separate from the ROB-125 source CTC moving-window decoding check.",
            "",
            f"Run manifest: `{args.run_manifest}`",
            f"Eval dir: `{args.eval_dir}`",
        ]
    )
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {"passed": passed, "primary_label": primary["run"]["label"] if primary else None}
    if args.pass_json_out:
        Path(args.pass_json_out).write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(args.summary_out)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
