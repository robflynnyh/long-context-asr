#!/usr/bin/env python3
"""Audit Floras label OOV rate for SentencePiece tokenizers.

This script reads the prepared Floras mapping and its aligned text JSON files,
counts label words, then reports how often each tokenizer emits its UNK id when
encoding those words. It intentionally does not load audio tensors.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import html
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, Tuple

import sentencepiece as spm


WORD_RE = re.compile(r"\S+")
SPEAKER_MARKER_RE = re.compile(r"(?<!\S)(?:>{1,}|<{1,})(?!\S)")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_EDGE_STRIP = "\"'`.,;:!?()[]{}<>"

TRANSLATION_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u02bc": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        default="/users/acp21rjf/align_floras50/tmp/mapping.json",
        help="Prepared Floras mapping JSON.",
    )
    parser.add_argument(
        "--tokenizer",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        default=[],
        help="Tokenizer name and SentencePiece model path. Can be repeated.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Use threads by default to avoid large transcript IPC overhead.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit.")
    parser.add_argument("--examples", type=int, default=50)
    parser.add_argument(
        "--normalization",
        choices=("none", "safe"),
        default="safe",
        help=(
            "On-the-fly label normalization to audit in addition to raw labels. "
            "'safe' recursively unescapes HTML, normalizes common Unicode punctuation, "
            "removes standalone speaker arrows, strips bracket characters while "
            "preserving their contents, trims leading/trailing token punctuation, "
            "drops tokens with no alphanumeric content, and collapses whitespace."
        ),
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def html_unescape_recursive(text: str, max_depth: int = 3) -> str:
    for _ in range(max_depth):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


def normalize_word(word: str) -> str:
    word = word.strip(TOKEN_EDGE_STRIP)
    word = word.strip("-")
    word = word.strip(TOKEN_EDGE_STRIP)
    return word


def has_alnum(text: str) -> bool:
    return any(char.isalnum() for char in text)


def normalize_safe(text: str) -> str:
    text = html_unescape_recursive(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TRANSLATION_TABLE)
    text = SPEAKER_MARKER_RE.sub(" ", text)
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("(", " ").replace(")", " ")
    words = [
        word
        for word in (normalize_word(word) for word in WORD_RE.findall(text))
        if word and has_alnum(word)
    ]
    return WHITESPACE_RE.sub(" ", " ".join(words)).strip()


def alnum_text(text: str) -> str:
    text = html_unescape_recursive(text)
    text = unicodedata.normalize("NFKC", text)
    return "".join(char.lower() for char in text if char.isalnum())


def read_record(item: Tuple[str, Dict[str, object], str]) -> Tuple[str, str, str, str, str | None]:
    record_id, record, normalization = item
    txt_path = str(record.get("txt", ""))
    if not txt_path or not os.path.exists(txt_path):
        return record_id, txt_path, "", "", "missing_txt"

    with open(txt_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    timestamps = payload.get("word_timestamps", [])
    words = [
        str(element.get("text", ""))
        for element in timestamps
        if str(element.get("text", "")).strip()
    ]
    raw_text = " ".join(words)
    normalized_text = normalize_safe(raw_text) if normalization == "safe" else raw_text
    return record_id, txt_path, raw_text, normalized_text, None


def describe_non_ascii(text: str) -> str:
    parts = []
    for char in text:
        if ord(char) > 127:
            parts.append(f"{char} U+{ord(char):04X} {unicodedata.name(char, '?')}")
    return "; ".join(parts)


def iter_records(mapping: Dict[str, Dict[str, object]], limit: int) -> Iterable[Tuple[str, Dict[str, object]]]:
    items = mapping.items()
    if limit and limit > 0:
        items = list(items)[:limit]
    return items


def empty_tokenizer_stats() -> Dict[str, object]:
    return {
        "word_counts": collections.Counter(),
        "first_seen": {},
        "record_words": {},
        "word_total": 0,
        "unique_words": 0,
    }


def update_text_stats(
    stats: Dict[str, object],
    record_id: str,
    txt_path: str,
    text: str,
) -> None:
    words = WORD_RE.findall(text)
    word_counts = stats["word_counts"]
    first_seen = stats["first_seen"]
    record_words = stats["record_words"]
    word_counts.update(words)
    unique_words = set(words)
    record_words[record_id] = unique_words
    for word in unique_words:
        first_seen.setdefault(word, (record_id, txt_path))


def empty_transcript_stats() -> Dict[str, object]:
    return {
        "changed_records": 0,
        "empty_after_normalization": 0,
        "records_with_alnum_loss": 0,
        "records_with_large_word_count_drop": 0,
        "raw_word_total": 0,
        "normalized_word_total": 0,
        "changed_examples": [],
        "alnum_loss_examples": [],
        "large_word_count_drop_examples": [],
    }


def update_transcript_stats(
    stats: Dict[str, object],
    record_id: str,
    txt_path: str,
    raw_text: str,
    normalized_text: str,
    max_examples: int,
) -> None:
    raw_words = WORD_RE.findall(raw_text)
    normalized_words = WORD_RE.findall(normalized_text)
    stats["raw_word_total"] += len(raw_words)
    stats["normalized_word_total"] += len(normalized_words)

    if raw_text != normalized_text:
        stats["changed_records"] += 1
        if len(stats["changed_examples"]) < max_examples:
            stats["changed_examples"].append(
                {
                    "record_id": record_id,
                    "txt": txt_path,
                    "raw": raw_text[:500],
                    "normalized": normalized_text[:500],
                }
            )

    if raw_words and not normalized_words:
        stats["empty_after_normalization"] += 1

    raw_alnum = alnum_text(raw_text)
    normalized_alnum = alnum_text(normalized_text)
    if raw_alnum != normalized_alnum:
        stats["records_with_alnum_loss"] += 1
        if len(stats["alnum_loss_examples"]) < max_examples:
            stats["alnum_loss_examples"].append(
                {
                    "record_id": record_id,
                    "txt": txt_path,
                    "raw_alnum": raw_alnum[:500],
                    "normalized_alnum": normalized_alnum[:500],
                    "raw": raw_text[:500],
                    "normalized": normalized_text[:500],
                }
            )

    if raw_words and len(normalized_words) < 0.9 * len(raw_words):
        stats["records_with_large_word_count_drop"] += 1
        if len(stats["large_word_count_drop_examples"]) < max_examples:
            stats["large_word_count_drop_examples"].append(
                {
                    "record_id": record_id,
                    "txt": txt_path,
                    "raw_word_count": len(raw_words),
                    "normalized_word_count": len(normalized_words),
                    "raw": raw_text[:500],
                    "normalized": normalized_text[:500],
                }
            )


def main() -> None:
    args = parse_args()
    tokenizers = args.tokenizer or [
        ("spotify_default", "/users/acp21rjf/long-context-asr/lcasr/artifacts/tokenizer.model"),
        ("floras50", "/users/acp21rjf/long-context-asr/lcasr/artifacts/floras50/tokenizer.model"),
    ]

    started = time.time()
    with open(args.mapping, "r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    records = list(iter_records(mapping, args.limit))

    print(f"mapping={args.mapping}", flush=True)
    print(f"records={len(records)} full_records={len(mapping)}", flush=True)
    print(f"workers={args.workers}", flush=True)
    print(f"executor={args.executor}", flush=True)
    print(f"normalization={args.normalization}", flush=True)

    processors = {}
    for name, path in tokenizers:
        processor = spm.SentencePieceProcessor(model_file=path)
        processors[name] = (path, processor)
        print(
            f"tokenizer[{name}]={path} vocab_size={processor.vocab_size()} "
            f"unk_id={processor.unk_id()} unk_piece={processor.id_to_piece(processor.unk_id())}",
            flush=True,
        )

    text_modes = ["raw"]
    if args.normalization != "none":
        text_modes.append(args.normalization)
    text_stats = {mode: empty_tokenizer_stats() for mode in text_modes}
    transcript_stats = empty_transcript_stats() if args.normalization != "none" else None
    missing_txt = 0
    empty_word_records = 0

    executor_class = (
        concurrent.futures.ThreadPoolExecutor
        if args.executor == "thread"
        else concurrent.futures.ProcessPoolExecutor
    )
    progress_interval = 500 if args.normalization != "none" else 5000
    with executor_class(max_workers=args.workers) as executor:
        futures = [
            executor.submit(read_record, (record_id, record, args.normalization))
            for record_id, record in records
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record_id, txt_path, raw_text, normalized_text, error = future.result()
            if error == "missing_txt":
                missing_txt += 1
            if not WORD_RE.findall(raw_text):
                empty_word_records += 1

            update_text_stats(text_stats["raw"], record_id, txt_path, raw_text)
            if args.normalization != "none":
                update_text_stats(text_stats[args.normalization], record_id, txt_path, normalized_text)
                update_transcript_stats(
                    transcript_stats,
                    record_id,
                    txt_path,
                    raw_text,
                    normalized_text,
                    args.examples,
                )

            if index % progress_interval == 0:
                print(
                    f"collect_progress={index}/{len(records)} "
                    f"raw_unique_words={len(text_stats['raw']['word_counts'])} "
                    f"elapsed_sec={time.time() - started:.1f}",
                    flush=True,
                )

    for mode, stats in text_stats.items():
        stats["word_total"] = sum(stats["word_counts"].values())
        stats["unique_words"] = len(stats["word_counts"])

    print(
        f"collection_done raw_word_total={text_stats['raw']['word_total']} "
        f"raw_unique_words={text_stats['raw']['unique_words']} missing_txt={missing_txt} "
        f"empty_word_records={empty_word_records} elapsed_sec={time.time() - started:.1f}",
        flush=True,
    )
    if transcript_stats is not None:
        print(
            "transcript_normalization "
            f"changed_records={transcript_stats['changed_records']} "
            f"records_with_alnum_loss={transcript_stats['records_with_alnum_loss']} "
            f"empty_after_normalization={transcript_stats['empty_after_normalization']} "
            f"large_word_count_drop={transcript_stats['records_with_large_word_count_drop']}",
            flush=True,
        )

    output = {
        "mapping": args.mapping,
        "records": len(records),
        "full_records": len(mapping),
        "workers": args.workers,
        "normalization": args.normalization,
        "word_total": text_stats["raw"]["word_total"],
        "unique_words": text_stats["raw"]["unique_words"],
        "missing_txt": missing_txt,
        "empty_word_records": empty_word_records,
        "transcript_normalization": transcript_stats,
        "tokenizers": {},
    }

    for name, (path, processor) in processors.items():
        unk_id = processor.unk_id()
        tokenizer_result = {
            "path": path,
            "vocab_size": processor.vocab_size(),
            "unk_id": unk_id,
            "unk_piece": processor.id_to_piece(unk_id),
            "modes": {},
        }

        for mode, stats in text_stats.items():
            word_counts = stats["word_counts"]
            first_seen = stats["first_seen"]
            record_words = stats["record_words"]
            word_total = stats["word_total"]
            subword_total = 0
            subword_unk = 0
            word_with_unk = 0
            oov_counts: collections.Counter[str] = collections.Counter()
            examples = {}

            for index, (word, count) in enumerate(word_counts.items(), 1):
                ids = processor.encode(word)
                unk_count = sum(1 for token in ids if token == unk_id)
                subword_total += len(ids) * count
                subword_unk += unk_count * count
                if unk_count:
                    word_with_unk += count
                    oov_counts[word] = count
                    if len(examples) < args.examples:
                        record_id, txt_path = first_seen[word]
                        examples[word] = {
                            "count": count,
                            "record_id": record_id,
                            "txt": txt_path,
                            "ids": ids,
                            "pieces": [processor.id_to_piece(token) for token in ids],
                            "non_ascii": describe_non_ascii(word),
                        }

                if index % 50000 == 0:
                    print(
                        f"encode_progress tokenizer={name} mode={mode} "
                        f"{index}/{len(word_counts)} elapsed_sec={time.time() - started:.1f}",
                        flush=True,
                    )

            oov_words = set(oov_counts)
            records_with_unk = sum(1 for words in record_words.values() if words & oov_words)
            unique_words = len(word_counts)
            unique_oov_words = len(oov_words)
            result = {
                "word_total": word_total,
                "word_with_unk": word_with_unk,
                "word_unk_rate": word_with_unk / word_total if word_total else 0.0,
                "unique_words": unique_words,
                "unique_oov_words": unique_oov_words,
                "unique_oov_rate": unique_oov_words / unique_words if unique_words else 0.0,
                "subword_total_wordwise": subword_total,
                "subword_unk_wordwise": subword_unk,
                "subword_unk_rate_wordwise": subword_unk / subword_total if subword_total else 0.0,
                "records_with_unk": records_with_unk,
                "record_unk_rate": records_with_unk / len(records) if records else 0.0,
                "top_oov_words": oov_counts.most_common(50),
                "examples": examples,
            }
            tokenizer_result["modes"][mode] = result

            print(f"RESULT tokenizer={name} mode={mode}", flush=True)
            for key in (
                "word_total",
                "word_with_unk",
                "word_unk_rate",
                "unique_words",
                "unique_oov_words",
                "unique_oov_rate",
                "subword_total_wordwise",
                "subword_unk_wordwise",
                "subword_unk_rate_wordwise",
                "records_with_unk",
                "record_unk_rate",
            ):
                print(f"  {key}={result[key]}", flush=True)
            print(
                "  top_oov_words="
                + json.dumps(result["top_oov_words"][:30], ensure_ascii=False),
                flush=True,
            )

        output["tokenizers"][name] = tokenizer_result

    output["elapsed_sec"] = time.time() - started
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote={output_path}", flush=True)
    print(f"elapsed_sec_total={output['elapsed_sec']:.1f}", flush=True)


if __name__ == "__main__":
    main()
