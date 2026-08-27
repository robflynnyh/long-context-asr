#!/usr/bin/env python3
"""ROB-327 Rev16 transcript capture for Floras-only CTC thesis checkpoints."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

import lcasr
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.buffered_transcription import fetch_logits as buffered_eval
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.general import get_model_class, load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))
import run as eval_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one Floras-only CTC thesis checkpoint on Rev16 and save "
            "structured transcripts for failure-case analysis."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["test", "test_long"])
    parser.add_argument("--seq-len", type=int, default=360000)
    parser.add_argument("--overlap", type=int, default=315000)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--override-window-size", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=3600000)
    parser.add_argument(
        "--evaluation-mode",
        default="windowed_attention",
        choices=["averaged_moving_window", "windowed_attention", "buffered"],
    )
    parser.add_argument("--model-class", default="SCConformerXL")
    parser.add_argument(
        "--tokenizer-path",
        default=str(REPO_ROOT / "lcasr" / "artifacts" / "floras50" / "tokenizer.model"),
    )
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def prepare_eval_args(args, model_config):
    seq_len = args.seq_len
    if args.evaluation_mode == "windowed_attention":
        model_config.model.attention_window_size = args.window_size
        seq_len = args.max_sequence_length
    if args.override_window_size is not None:
        model_config.model.attention_window_size = args.override_window_size

    return argparse.Namespace(
        checkpoint=args.checkpoint,
        split=args.split,
        seq_len=seq_len,
        overlap=args.overlap,
        config=model_config,
        model_class=args.model_class,
        evaluation_mode=args.evaluation_mode,
        tokenizer_path=args.tokenizer_path,
        overide_window_size=args.override_window_size,
    )


def select_eval_fn(args):
    if args.evaluation_mode == "buffered":
        return buffered_eval
    return moving_average_eval


def load_checkpoint_and_model(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    eval_args = prepare_eval_args(args, model_config)

    tokenizer_kwargs = {}
    if args.tokenizer_path:
        tokenizer_kwargs["tokenizer_path"] = args.tokenizer_path
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)

    model_class = get_model_class({"model_class": model_config.get("model_class", args.model_class)})
    model = load_model(model_config, tokenizer.vocab_size(), model_class=model_class)
    model.print_total_params()
    model.load_state_dict(checkpoint["model"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.device = device
    model = model.to(device)
    model.eval()
    return model, tokenizer, eval_args, device


def filter_records(data, args):
    if args.recording_id:
        wanted = set(args.recording_id)
        data = [record for record in data if record["id"] in wanted]
        missing = sorted(wanted - {record["id"] for record in data})
        if missing:
            raise ValueError(f"Requested Rev16 recording ids not found: {missing}")
    if args.max_recordings is not None:
        data = data[: args.max_recordings]
    return data


def wer_record(hypothesis, reference):
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
        hypotheses=[hypothesis],
        references=[reference],
    )
    cer, chars, char_ins_rate, char_del_rate, char_sub_rate = word_error_rate_detail(
        hypotheses=[hypothesis],
        references=[reference],
        use_cer=True,
    )
    return {
        "wer": wer,
        "cer": cer,
        "words": words,
        "chars": chars,
        "ins_rate": ins_rate,
        "del_rate": del_rate,
        "sub_rate": sub_rate,
        "char_ins_rate": char_ins_rate,
        "char_del_rate": char_del_rate,
        "char_sub_rate": char_sub_rate,
        "hyp_words": len(hypothesis.split()),
        "ref_words": len(reference.split()),
        "hyp_chars": len(hypothesis.replace(" ", "")),
        "ref_chars": len(reference.replace(" ", "")),
    }


def write_outputs(output_dir, records, summary):
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / "transcripts.jsonl"
    with transcript_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    metric_keys = [
        "recording",
        "wer",
        "cer",
        "words",
        "chars",
        "ins_rate",
        "del_rate",
        "sub_rate",
        "char_ins_rate",
        "char_del_rate",
        "char_sub_rate",
        "hyp_words",
        "ref_words",
        "hyp_chars",
        "ref_chars",
    ]
    with (output_dir / "per_record_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in metric_keys})

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    model, tokenizer, eval_args, device = load_checkpoint_and_model(args)
    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    eval_fn = select_eval_fn(args)

    data = eval_run.get_dataset_function("rev16")(args.split)
    data = filter_records(data, args)
    if not data:
        raise ValueError("No Rev16 records selected for evaluation")

    records = []
    hypotheses = []
    references = []
    for index, item in enumerate(data):
        if not args.quiet:
            print(f"Processing {index + 1}/{len(data)}: {item['id']}", flush=True)

        audio_spec, reference = item["process_fn"](item)
        logits = eval_fn(
            args=eval_args,
            model=model,
            spec=audio_spec,
            seq_len=eval_args.seq_len,
            overlap=eval_args.overlap,
            tokenizer=tokenizer,
        )
        hypothesis = eval_run.normalize(decoder(torch.as_tensor(logits))).lower().strip()
        reference = reference.strip()

        record = {
            "recording": item["id"],
            "split": args.split,
            "audio": item["audio"],
            "reference": reference,
            "hypothesis": hypothesis,
            **wer_record(hypothesis, reference),
        }
        records.append(record)
        hypotheses.append(hypothesis)
        references.append(reference)

    aggregate = wer_record(" ".join(hypotheses).strip(), " ".join(references).strip())
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "records": len(records),
        "evaluation_mode": args.evaluation_mode,
        "seq_len_arg": args.seq_len,
        "effective_seq_len": eval_args.seq_len,
        "overlap": args.overlap,
        "window_size": args.window_size,
        "override_window_size": args.override_window_size,
        "max_sequence_length": args.max_sequence_length,
        "tokenizer_path": args.tokenizer_path,
        "device": str(device),
        **aggregate,
    }

    write_outputs(output_dir, records, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
