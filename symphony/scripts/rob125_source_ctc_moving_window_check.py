#!/usr/bin/env python3
import argparse
import csv
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
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.general import get_model_class, load_model
from tedlium.run import get_text_and_audio


normalize = EnglishTextNormalizer()


class SourceCTCView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decoder = model.ctc_decoder
        self.subsampling = model.subsampling
        self.device = getattr(model, "device", None)

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return {
            "final_posteriors": output["final_posteriors_ctc"],
            "length": output["length"],
        }


def length_diagnostics(hypotheses, references):
    return {
        "hyp_words": sum(len(text.split()) for text in hypotheses),
        "ref_words": sum(len(text.split()) for text in references),
        "hyp_chars": sum(len(text.replace(" ", "")) for text in hypotheses),
        "ref_chars": sum(len(text.replace(" ", "")) for text in references),
    }


def score_rows(hypotheses, references, recordings):
    rows = []
    for recording, hypothesis, reference in zip(recordings, hypotheses, references):
        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail([hypothesis], [reference])
        cer, chars, char_ins_rate, char_del_rate, char_sub_rate = word_error_rate_detail(
            [hypothesis], [reference], use_cer=True
        )
        rows.append(
            {
                "recording": recording,
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
                **length_diagnostics([hypothesis], [reference]),
            }
        )

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses, references)
    cer, chars, char_ins_rate, char_del_rate, char_sub_rate = word_error_rate_detail(
        hypotheses, references, use_cer=True
    )
    rows.append(
        {
            "recording": "all",
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
            **length_diagnostics(hypotheses, references),
        }
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--disable-flash-attention", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "source_ctc_moving_window_results.csv"
    jsonl_path = output_dir / "source_ctc_moving_window_predictions.jsonl"

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    if args.disable_flash_attention:
        config.model.flash_attn = False

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(config, tokenizer.vocab_size(), model_class=get_model_class(config=config))
    model.load_state_dict(checkpoint["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.device = device
    model = model.to(device).eval()

    ctc_model = SourceCTCView(model).to(device).eval()
    ctc_model.device = device
    decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=ctc_model.decoder.num_classes - 1)

    eval_args = argparse.Namespace(config=config)
    data = get_text_and_audio(args.split)
    if args.max_records is not None:
        data = data[: args.max_records]

    hypotheses, references, recordings = [], [], []
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for rec in data:
            audio_spec, reference = rec["process_fn"](rec)
            logits = moving_average_eval(
                args=eval_args,
                model=ctc_model,
                spec=audio_spec,
                seq_len=args.seq_len,
                overlap=args.overlap,
                tokenizer=tokenizer,
                use_tqdm=args.verbose,
            )
            hypothesis = normalize(decoder(torch.as_tensor(logits))).lower().strip()
            reference = reference.strip()
            hypotheses.append(hypothesis)
            references.append(reference)
            recordings.append(rec["id"])
            jsonl.write(
                json.dumps(
                    {
                        "recording": rec["id"],
                        "reference": reference,
                        "hypothesis": hypothesis,
                    }
                )
                + "\n"
            )
            if args.verbose:
                print(rec["id"])
                print(reference)
                print(hypothesis)

    rows = score_rows(hypotheses, references, recordings)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = rows[-1]
    print(json.dumps({"csv": str(csv_path), "predictions": str(jsonl_path), **aggregate}, indent=2))


if __name__ == "__main__":
    main()
