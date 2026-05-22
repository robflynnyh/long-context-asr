#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percent(value):
    return f"{float(value) * 100:.2f}%"


def last_jsonl(path):
    if not path or not Path(path).exists():
        return None
    last = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.run_manifest).read_text(encoding="utf-8"))
    result = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    train_diag = last_jsonl(manifest.get("diagnostics_path"))
    overfit_pass = result["wer"] <= 0.20 and result["hyp_words"] > 0

    lines = [
        "# ROB-128 One-Record Probe Overfit Check",
        "",
        f"Record: `{manifest['record_id']}`",
        f"Source checkpoint: `{manifest['source_checkpoint']}`",
        f"Trained checkpoint: `{result['checkpoint']}`",
        f"Head: `{manifest['head']}`",
        f"Epochs requested: `{manifest['max_epochs']}`",
        f"Learning rate: `{manifest['learning_rate']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| WER | {percent(result['wer'])} |",
        f"| CER | {percent(result['cer'])} |",
        f"| Insertions | {percent(result['ins_rate'])} |",
        f"| Deletions | {percent(result['del_rate'])} |",
        f"| Substitutions | {percent(result['sub_rate'])} |",
        f"| Hyp words | {result['hyp_words']} |",
        f"| Ref words | {result['ref_words']} |",
        f"| Eval CTC loss | {result['eval_ctc_loss']:.4f} |",
        f"| Eval blank p | {percent(result['blank_p'])} |",
        f"| Input frames | {result['input_frames']} |",
        f"| Output frames | {result['output_frames']} |",
        f"| Target tokens | {result['target_tokens']} |",
        f"| Nonempty reference chunks | {result['nonempty_reference_chunks']} / {result['reference_chunks']} |",
    ]
    if train_diag:
        lines.extend(
            [
                f"| Final train logged loss | {train_diag['loss']:.4f} |",
                f"| Final train logged blank p | {percent(train_diag['blank_p'])} |",
            ]
        )
    lines.extend(
        [
            "",
            f"Overfit status: `{'pass' if overfit_pass else 'fail'}`",
            "",
            "Interpretation:",
        ]
    )
    if overfit_pass:
        lines.append(
            "The one-record supervised-feature probe can overfit enough to emit useful words. "
            "The next ROB-128 step is the requested head/initialization ablation."
        )
    else:
        lines.append(
            "The one-record supervised-feature probe did not overfit to a useful transcript. "
            "Debug CTC targets/lengths, hidden-state normalization, trainable parameters, "
            "and checkpoint/eval loading before interpreting ROB-100 frozen-probe WER."
        )
    lines.extend(
        [
            "",
            "Hypothesis excerpt:",
            "```text",
            result["hypothesis_normalized"][:1200],
            "```",
            "",
            "Reference excerpt:",
            "```text",
            result["reference_normalized"][:1200],
            "```",
            "",
            f"Run manifest: `{args.run_manifest}`",
            f"Eval JSON: `{args.eval_json}`",
        ]
    )
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_out)


if __name__ == "__main__":
    main()
