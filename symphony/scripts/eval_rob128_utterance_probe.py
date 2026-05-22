#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from whisper.normalizers import EnglishTextNormalizer

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import lcasr
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.general import get_model_class, load_model


normalize = EnglishTextNormalizer()


class SourceCTCView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.ctc_decoder
        self.subsampling = model.subsampling

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return {
            "final_posteriors": output["final_posteriors_ctc"],
            "length": output["length"],
        }


def latest_checkpoint(checkpoint_dir):
    paths = sorted(Path(checkpoint_dir).glob("step_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    if not paths:
        raise SystemExit(f"no checkpoints found in {checkpoint_dir}")
    return paths[-1]


def load_training_checkpoint(path, tokenizer, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = load_model(config, tokenizer.vocab_size(), model_class=get_model_class(config=config))
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing:
        print(f"missing keys after probe checkpoint load: {len(missing)}")
    if unexpected:
        print(f"unexpected keys after probe checkpoint load: {len(unexpected)}")
    model = model.to(device).eval()
    model.device = device
    return model, config


def load_source_ctc_model(path, tokenizer, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = load_model(config, tokenizer.vocab_size(), model_class=get_model_class(config=config))
    model.load_state_dict(checkpoint["model"], strict=False)
    model = SourceCTCView(model).to(device).eval()
    model.device = device
    return model, config


def batch_files(files):
    items = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    ids = [item["id"] for item in items]
    texts = [item.get("text", "") for item in items]
    audio_lengths = torch.cat([item["audio_lengths"] for item in items])
    txt_lengths = torch.cat([item["txt_lengths"] for item in items])
    max_audio = int(audio_lengths.max())
    max_txt = int(txt_lengths.max())
    audio = torch.cat(
        [
            torch.nn.functional.pad(item["audio"], (0, max_audio - int(item["audio_lengths"][0])), value=0)
            for item in items
        ],
        dim=0,
    )
    txt = torch.cat(
        [
            torch.nn.functional.pad(item["txt"], (0, max_txt - int(item["txt_lengths"][0])), value=0)
            for item in items
        ],
        dim=0,
    )
    return ids, texts, audio, audio_lengths, txt, txt_lengths


def score(hypotheses, references):
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses, references)
    cer, chars, char_ins_rate, char_del_rate, char_sub_rate = word_error_rate_detail(
        hypotheses, references, use_cer=True
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
        "hyp_words": sum(len(text.split()) for text in hypotheses),
        "ref_words": sum(len(text.split()) for text in references),
        "hyp_chars": sum(len(text.replace(" ", "")) for text in hypotheses),
        "ref_chars": sum(len(text.replace(" ", "")) for text in references),
    }


def evaluate_model(model, tokenizer, utterance_dir, batch_size, device):
    files = sorted(Path(utterance_dir).glob("*.pt"))
    if not files:
        raise SystemExit(f"no utterance .pt files found in {utterance_dir}")
    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    ctc_loss = torch.nn.CTCLoss(blank=model.decoder.num_classes - 1, reduction="sum", zero_infinity=True)
    hypotheses = []
    references = []
    utterance_rows = []
    total_loss = 0.0
    total_items = 0
    blank_frames = 0
    total_frames = 0
    input_lengths = []
    output_lengths = []
    target_lengths = []

    for start in range(0, len(files), batch_size):
        ids, texts, audio, audio_lengths, txt, txt_lengths = batch_files(files[start : start + batch_size])
        audio = audio.to(device)
        audio_lengths = audio_lengths.to(device)
        txt = txt.to(device)
        txt_lengths = txt_lengths.to(device)
        with torch.no_grad():
            out = model(audio_signal=audio, length=audio_lengths)
            log_probs = out["final_posteriors"]
            lengths = out["length"]
            loss = ctc_loss(log_probs.transpose(0, 1), txt, lengths, txt_lengths)
        preds = log_probs.detach().cpu().argmax(dim=-1)
        blank_id = log_probs.shape[-1] - 1
        blank_frames += int((preds == blank_id).sum().item())
        total_frames += int(preds.numel())
        total_loss += float(loss.item())
        total_items += len(ids)
        input_lengths.extend(int(x) for x in audio_lengths.cpu().tolist())
        output_lengths.extend(int(x) for x in lengths.cpu().tolist())
        target_lengths.extend(int(x) for x in txt_lengths.cpu().tolist())

        for idx, utt_id in enumerate(ids):
            hyp = normalize(decoder(log_probs[idx].detach().cpu())).lower().strip()
            ref = normalize(texts[idx]).lower().strip()
            hypotheses.append(hyp)
            references.append(ref)
            utterance_rows.append(
                {
                    "id": utt_id,
                    "reference": ref,
                    "hypothesis": hyp,
                    "input_length": int(audio_lengths[idx].cpu().item()),
                    "output_length": int(lengths[idx].cpu().item()),
                    "target_length": int(txt_lengths[idx].cpu().item()),
                }
            )

    metrics = score(hypotheses, references)
    metrics.update(
        {
            "ctc_loss_sum": total_loss,
            "ctc_loss_per_utterance": total_loss / max(total_items, 1),
            "blank_rate": blank_frames / max(total_frames, 1),
            "utterances": total_items,
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
            "target_lengths": target_lengths,
            "greedy_transcript": " ".join(hypotheses).strip(),
            "reference_transcript": " ".join(references).strip(),
            "utterance_predictions": utterance_rows,
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterance-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--source-ctc-checkpoint")
    args = parser.parse_args()

    if bool(args.source_ctc_checkpoint) == bool(args.checkpoint or args.checkpoint_dir):
        raise SystemExit("provide exactly one of --source-ctc-checkpoint or --checkpoint/--checkpoint-dir")
    checkpoint = args.checkpoint
    if args.checkpoint_dir:
        checkpoint = str(latest_checkpoint(args.checkpoint_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    if args.source_ctc_checkpoint:
        model, config = load_source_ctc_model(args.source_ctc_checkpoint, tokenizer, device)
        checkpoint_path = args.source_ctc_checkpoint
    else:
        model, config = load_training_checkpoint(checkpoint, tokenizer, device)
        checkpoint_path = checkpoint

    result = evaluate_model(model, tokenizer, args.utterance_dir, args.batch_size, device)
    result.update(
        {
            "label": args.label,
            "checkpoint": checkpoint_path,
            "utterance_dir": args.utterance_dir,
            "model_class": config.get("model_class", "unknown"),
        }
    )
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
