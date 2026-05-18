import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torchaudio
from tqdm import tqdm

import lcasr
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import get_model_class, load_model

try:
    from whisper.normalizers import EnglishTextNormalizer
except Exception:
    EnglishTextNormalizer = None


DEFAULT_TEDLIUM_ROOT = "/store/store4/data/TEDLIUM_release1/legacy"


def normalize_text(text: str) -> str:
    if EnglishTextNormalizer is not None:
        text = EnglishTextNormalizer()(text)
    text = text.lower().strip()
    text = text[:-1].strip() if text.endswith(".") else text
    return re.sub(r" +", " ", text)


def clean_stm_text(text: str) -> str:
    text = re.sub(r" '([a-z])", r"'\1", text).strip()
    return re.sub(r" +", " ", text)


def iter_stm_utterances(stm_path: Path) -> Iterable[Dict[str, Any]]:
    with stm_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            parts = line.strip().split(" ")
            if len(parts) < 7:
                continue
            recording_id, _, speaker, start, end, _ = parts[:6]
            text = clean_stm_text(" ".join(parts[6:]))
            if text == "ignore_time_segment_in_scoring":
                continue
            yield {
                "recording_id": recording_id,
                "utterance_id": f"{recording_id}:{line_index}",
                "speaker": speaker,
                "start": float(start),
                "end": float(end),
                "reference": text,
            }


def fetch_tedlium_items(root: Path, split: str, max_recordings: Optional[int]) -> List[Dict[str, Any]]:
    split_root = root / split
    audio_root = split_root / "sph"
    stm_root = split_root / "stm"
    if not audio_root.is_dir() or not stm_root.is_dir():
        raise FileNotFoundError(f"Expected TEDLIUM split dirs under {split_root}: sph/ and stm/")

    audio_paths = sorted(audio_root.glob("*.sph"))
    if max_recordings is not None:
        audio_paths = audio_paths[:max_recordings]

    items: List[Dict[str, Any]] = []
    for audio_path in audio_paths:
        stm_path = stm_root / f"{audio_path.stem}.stm"
        if not stm_path.exists():
            continue
        for utterance in iter_stm_utterances(stm_path):
            items.append({**utterance, "audio_path": str(audio_path), "stm_path": str(stm_path)})
    return items


def decode_prediction_ids(tokenizer: Any, prediction_ids: Iterable[int], silence_id: int, max_tokens: int) -> str:
    tokens: List[int] = []
    previous: Optional[int] = None
    for idx in prediction_ids:
        idx = int(idx)
        if idx == silence_id:
            previous = idx
            continue
        if idx == previous:
            continue
        tokens.append(idx)
        previous = idx
        if len(tokens) >= max_tokens:
            break
    return "" if not tokens else tokenizer.decode(tokens)


def audio_frame_window(path: str, start: float, end: float) -> Dict[str, int]:
    info = torchaudio.info(path)
    sample_rate = int(info.sample_rate)
    start_sample = max(int(start * sample_rate), 0)
    num_samples = max(int((end - start) * sample_rate), 1)
    return {"frame_offset": start_sample, "num_frames": num_samples}


def load_checkpoint_model(checkpoint_path: str, device: torch.device, dtype: Optional[torch.dtype]):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    tokenizer_kwargs = {}
    tokenizer_path = config.get("training", {}).get("tokenizer_path")
    if tokenizer_path is not None:
        tokenizer_kwargs["tokenizer_path"] = tokenizer_path
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)
    model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
    model.load_state_dict(checkpoint["model"])
    if dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer, config


def select_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    if name == "float32" or device.type == "cpu":
        return None
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def run_eval(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = select_dtype(args.dtype, device)
    model, tokenizer, config = load_checkpoint_model(args.checkpoint, device=device, dtype=dtype)
    silence_id = model.get_silence_id()

    items = fetch_tedlium_items(Path(args.tedlium_root), args.split, args.max_recordings)
    if args.max_utterances is not None:
        items = items[: args.max_utterances]
    if not items:
        raise RuntimeError("No TEDLIUM utterances selected for evaluation")

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json = Path(args.summary_json) if args.summary_json else None
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
    report_md = Path(args.report_md) if args.report_md else None
    if report_md is not None:
        report_md.parent.mkdir(parents=True, exist_ok=True)

    hypotheses: List[str] = []
    references: List[str] = []
    records: List[Dict[str, Any]] = []

    with output_jsonl.open("w", encoding="utf-8") as out:
        iterator = tqdm(items, total=len(items), disable=args.no_progress)
        for item in iterator:
            window = audio_frame_window(item["audio_path"], item["start"], item["end"])
            spectrogram = processing_chain(item["audio_path"], **window)
            if spectrogram.dim() == 3 and spectrogram.size(0) == 1:
                spectrogram = spectrogram.squeeze(0)
            spectrogram = spectrogram.to(device)
            if dtype is not None:
                spectrogram = spectrogram.to(dtype=dtype)
            length = torch.tensor([spectrogram.shape[-1]], dtype=torch.long, device=device)
            audio_signal = spectrogram.unsqueeze(0)

            if args.decode_mode == "sample":
                decoded = model.sample_decode(
                    audio_signal=audio_signal,
                    length=length,
                    max_frames=args.max_output_frames,
                    temperature=args.temperature,
                )
            else:
                decoded = model.greedy_decode(
                    audio_signal=audio_signal,
                    length=length,
                    max_frames=args.max_output_frames,
                )

            pred_len = int(decoded["length"][0].item())
            prediction_ids = decoded["predictions"][0, :pred_len].detach().cpu().tolist()
            raw_prediction = decode_prediction_ids(tokenizer, prediction_ids, silence_id, args.max_tokens)
            hypothesis = normalize_text(raw_prediction)
            reference = normalize_text(item["reference"])
            pred_non_silence_fraction = (
                sum(int(idx) != silence_id for idx in prediction_ids) / max(len(prediction_ids), 1)
            )

            record = {
                "utterance_id": item["utterance_id"],
                "recording_id": item["recording_id"],
                "speaker": item["speaker"],
                "start": item["start"],
                "end": item["end"],
                "duration": item["end"] - item["start"],
                "reference": reference,
                "prediction": hypothesis,
                "raw_prediction": raw_prediction,
                "output_frames": pred_len,
                "pred_non_silence_fraction": pred_non_silence_fraction,
                "audio_path": item["audio_path"],
                "stm_path": item["stm_path"],
            }
            out.write(json.dumps(record, ensure_ascii=True) + "\n")
            records.append(record)
            hypotheses.append(hypothesis)
            references.append(reference)

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=hypotheses, references=references)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "tedlium_root": args.tedlium_root,
        "split": args.split,
        "decode_mode": args.decode_mode,
        "temperature": args.temperature if args.decode_mode == "sample" else None,
        "device": str(device),
        "dtype": args.dtype if dtype is not None else "float32",
        "max_recordings": args.max_recordings,
        "max_utterances": args.max_utterances,
        "utterances": len(records),
        "wer": wer,
        "words": words,
        "ins_rate": ins_rate,
        "del_rate": del_rate,
        "sub_rate": sub_rate,
        "mean_pred_non_silence_fraction": sum(r["pred_non_silence_fraction"] for r in records) / len(records),
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json) if summary_json is not None else None,
        "model_class": config.get("model_class"),
        "subsampling_factor": config.get("model", {}).get("subsampling_factor"),
    }
    if summary_json is not None:
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_md is not None:
        report_md.write_text(render_report(summary, records, args), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def render_report(summary: Dict[str, Any], records: List[Dict[str, Any]], args) -> str:
    command = " ".join(args.command)
    lines = [
        "# ROB-90 Streaming Decoder TEDLIUM Utterance Eval",
        "",
        "This is a bounded wiring sanity check for the ROB-76 decoder-only streaming ASR checkpoint. Poor recognition quality is expected; the goal is to prove TEDLIUM utterance decoding runs and to inspect representative outputs.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Configuration",
        "",
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- TEDLIUM root: `{summary['tedlium_root']}`",
        f"- split: `{summary['split']}`",
        f"- decode mode: `{summary['decode_mode']}`",
        f"- utterances: `{summary['utterances']}`",
        f"- output JSONL: `{summary['output_jsonl']}`",
        f"- summary JSON: `{summary['summary_json']}`",
        f"- device/dtype: `{summary['device']}` / `{summary['dtype']}`",
        "",
        "## Summary",
        "",
        f"- WER: `{summary['wer']:.6f}`",
        f"- words: `{summary['words']}`",
        f"- insertions/deletions/substitutions: `{summary['ins_rate']:.6f}` / `{summary['del_rate']:.6f}` / `{summary['sub_rate']:.6f}`",
        f"- mean predicted non-silence fraction: `{summary['mean_pred_non_silence_fraction']:.6f}`",
        "",
        "## Sample Outputs",
        "",
        "| utterance | reference | prediction | pred non-silence |",
        "| --- | --- | --- | --- |",
    ]
    for record in records[: args.report_samples]:
        prediction = record["prediction"].replace("|", "\\|") or "<empty>"
        reference = record["reference"].replace("|", "\\|")
        lines.append(
            f"| `{record['utterance_id']}` | {reference} | {prediction} | {record['pred_non_silence_fraction']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate StreamingDecoderASR on TEDLIUM utterances.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tedlium-root", default=DEFAULT_TEDLIUM_ROOT)
    parser.add_argument("--split", choices=["dev", "test", "train"], default="test")
    parser.add_argument("--max-recordings", type=int, default=1)
    parser.add_argument("--max-utterances", type=int, default=8)
    parser.add_argument("--decode-mode", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-output-frames", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--report-md", default="")
    parser.add_argument("--report-samples", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    args.command = ["PYTHONPATH=.", "python", "eval/tedlium/run_streaming_decoder_asr.py"] + os.sys.argv[1:]
    run_eval(args)


if __name__ == "__main__":
    main()
