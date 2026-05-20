import argparse
import json
import sys
from pathlib import Path

import torch

import lcasr
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.general import get_model_class, load_model


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval" / "tedlium"))
import run as tedlium_run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Issue-local ROB-90 TEDLIUM utterance sanity eval for StreamingDecoderASR."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tedlium-root", required=True)
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-utterances", type=int, default=8)
    parser.add_argument("--recording-index", type=int, default=0)
    parser.add_argument("--decode-mode", default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-frames", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--eval-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return parser.parse_args()


def cuda_dtype(name, device):
    if device.type != "cuda" or name == "float32":
        return None
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def normalize(text):
    return tedlium_run.normalize(text).lower().strip()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(config, tokenizer.vocab_size(), model_class=get_model_class(config=config))
    model.load_state_dict(checkpoint["model"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = cuda_dtype(args.eval_dtype, device)
    model = model.to(device=device, dtype=dtype) if dtype is not None else model.to(device=device)
    model.eval()

    data_path = str(Path(args.tedlium_root) / args.split)
    audio_files, text_files = tedlium_run.fetch_data(path=data_path)
    if args.recording_index >= len(audio_files):
        raise IndexError(f"recording-index {args.recording_index} out of range for {len(audio_files)} recordings")

    audio_path = audio_files[args.recording_index]
    stm_path = text_files[args.recording_index]
    audio_spec = tedlium_run.processing_chain(audio_path)
    utterances, _ = tedlium_run.fetch_utterances(stm_path, audio_spec)
    if args.max_utterances is not None:
        utterances = utterances[: args.max_utterances]

    records = []
    references = []
    predictions = []
    recording_id = Path(audio_path).stem
    for index, utterance in enumerate(utterances):
        result = model.transcribe(
            utterance["spectogram"],
            tokenizer,
            device=device,
            decode_mode=args.decode_mode,
            temperature=args.temperature,
            max_output_frames=args.max_output_frames,
            max_tokens=args.max_tokens,
            return_metadata=True,
        )
        reference = normalize(utterance["text"])
        prediction = normalize(result["text"])
        records.append(
            {
                "utterance": f"{recording_id}:{index}",
                "start": utterance["start"],
                "end": utterance["end"],
                "reference": reference,
                "prediction": prediction,
                "output_frames": result["output_frames"],
                "pred_non_silence_fraction": result["pred_non_silence_fraction"],
            }
        )
        references.append(reference)
        predictions.append(prediction)

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
        hypotheses=[" ".join(predictions).strip()],
        references=[" ".join(references).strip()],
    )
    summary = {
        "checkpoint": args.checkpoint,
        "tedlium_root": args.tedlium_root,
        "split": args.split,
        "audio": audio_path,
        "stm": stm_path,
        "recording_index": args.recording_index,
        "utterances": len(records),
        "decode_mode": args.decode_mode,
        "eval_dtype": args.eval_dtype if device.type == "cuda" else "float32",
        "wer": wer,
        "words": words,
        "ins_rate": ins_rate,
        "del_rate": del_rate,
        "sub_rate": sub_rate,
        "mean_pred_non_silence_fraction": sum(
            record["pred_non_silence_fraction"] for record in records
        )
        / max(len(records), 1),
    }

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    for record in records:
        print(json.dumps(record, ensure_ascii=True))


if __name__ == "__main__":
    main()
