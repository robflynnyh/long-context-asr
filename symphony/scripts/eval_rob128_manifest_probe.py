#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from whisper.normalizers import EnglishTextNormalizer

import lcasr
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.utils import fetch_logits
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.dataloading import chunk_text_json
from lcasr.utils.general import get_model_class, load_model


normalize = EnglishTextNormalizer()


def latest_checkpoint(path: str):
    checkpoints = sorted(Path(path).glob("step_*.pt"), key=lambda item: int(item.stem.split("_")[1]))
    if not checkpoints:
        raise SystemExit(f"no step_*.pt checkpoints found in {path}")
    return checkpoints[-1]


def load_record_text(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "word_timestamps" in payload:
        return sorted(payload["word_timestamps"], key=lambda item: (float(item["start"]), float(item["end"])))
    return payload["results"][-1]["alternatives"][0]["words"]


def manifest_reference(entry: dict, spec_frames: int, seq_len: int):
    text_items = load_record_text(entry["txt"])
    chunks = chunk_text_json(text_items, chunk_size=seq_len, chunk_overlap=0, spectogram_length=spec_frames)
    raw_reference = " ".join(chunk for chunk in chunks if chunk.strip()).strip()
    return raw_reference, chunks


def score(hypothesis: str, reference: str):
    hyp_norm = normalize(hypothesis).lower().strip()
    ref_norm = normalize(reference).lower().strip()
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail([hyp_norm], [ref_norm])
    cer, chars, char_ins_rate, char_del_rate, char_sub_rate = word_error_rate_detail(
        [hyp_norm], [ref_norm], use_cer=True
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
        "hyp_words": len(hyp_norm.split()),
        "ref_words": len(ref_norm.split()),
        "hyp_chars": len(hyp_norm.replace(" ", "")),
        "ref_chars": len(ref_norm.replace(" ", "")),
        "hypothesis_normalized": hyp_norm,
        "reference_normalized": ref_norm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--disable-flash-attention", action="store_true")
    args = parser.parse_args()

    run_manifest = json.loads(Path(args.run_manifest).read_text(encoding="utf-8"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.checkpoint_dir or run_manifest["checkpoint_dir"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if args.disable_flash_attention and "model" in config:
        config.model.flash_attn = False
    seq_len = args.seq_len or int(run_manifest.get("seq_len", 2048))

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(config, tokenizer.vocab_size(), model_class=get_model_class(config=config))
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.device = device
    model = model.to(device).eval()

    entry = run_manifest["record"]
    spec = torch.load(entry["audio"], map_location="cpu")
    if spec.ndim == 2:
        spec = spec.unsqueeze(0)
    spec_frames = int(spec.shape[-1])
    reference, ref_chunks = manifest_reference(entry, spec_frames, seq_len)

    logits = torch.as_tensor(
        fetch_logits(
            args=argparse.Namespace(config=config),
            model=model,
            spec=spec,
            seq_len=seq_len,
            overlap=args.overlap,
            tokenizer=tokenizer,
            use_tqdm=False,
        )
    )
    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=logits.shape[-1] - 1)
    hypothesis = decoder(logits)
    target = torch.LongTensor(tokenizer.encode(reference))
    ctc_loss = torch.nn.CTCLoss(blank=logits.shape[-1] - 1, reduction="sum", zero_infinity=True)(
        logits.log_softmax(dim=-1).unsqueeze(1),
        target.unsqueeze(0),
        torch.LongTensor([logits.shape[0]]),
        torch.LongTensor([target.numel()]),
    )
    blank_p = float((logits.argmax(dim=-1) == logits.shape[-1] - 1).float().mean().item())

    result = {
        "checkpoint": str(checkpoint_path),
        "record_id": run_manifest["record_id"],
        "record_audio": entry["audio"],
        "record_txt": entry["txt"],
        "input_frames": spec_frames,
        "output_frames": int(logits.shape[0]),
        "target_tokens": int(target.numel()),
        "nonempty_reference_chunks": sum(bool(chunk.strip()) for chunk in ref_chunks),
        "reference_chunks": len(ref_chunks),
        "eval_ctc_loss": float(ctc_loss.item()),
        "blank_p": blank_p,
        "missing_state_keys": len(missing),
        "unexpected_state_keys": len(unexpected),
        "hypothesis": hypothesis,
        "reference": reference,
        **score(hypothesis, reference),
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
