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
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sentencepiece as spm


WORD_RE = re.compile(r"\S+")


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
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit.")
    parser.add_argument("--examples", type=int, default=50)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def read_record(item: Tuple[str, Dict[str, object]]) -> Tuple[str, str, List[str], str | None]:
    record_id, record = item
    txt_path = str(record.get("txt", ""))
    if not txt_path or not os.path.exists(txt_path):
        return record_id, txt_path, [], "missing_txt"

    with open(txt_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    timestamps = payload.get("word_timestamps", [])
    words = [
        str(element.get("text", ""))
        for element in timestamps
        if str(element.get("text", "")).strip()
    ]
    return record_id, txt_path, WORD_RE.findall(" ".join(words)), None


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

    processors = {}
    for name, path in tokenizers:
        processor = spm.SentencePieceProcessor(model_file=path)
        processors[name] = (path, processor)
        print(
            f"tokenizer[{name}]={path} vocab_size={processor.vocab_size()} "
            f"unk_id={processor.unk_id()} unk_piece={processor.id_to_piece(processor.unk_id())}",
            flush=True,
        )

    word_counts: collections.Counter[str] = collections.Counter()
    first_seen: Dict[str, Tuple[str, str]] = {}
    record_words: Dict[str, set[str]] = {}
    missing_txt = 0
    empty_word_records = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(read_record, item) for item in records]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record_id, txt_path, words, error = future.result()
            if error == "missing_txt":
                missing_txt += 1
            if not words:
                empty_word_records += 1

            word_counts.update(words)
            unique_words = set(words)
            record_words[record_id] = unique_words
            for word in unique_words:
                first_seen.setdefault(word, (record_id, txt_path))

            if index % 5000 == 0:
                print(
                    f"collect_progress={index}/{len(records)} "
                    f"unique_words={len(word_counts)} elapsed_sec={time.time() - started:.1f}",
                    flush=True,
                )

    print(
        f"collection_done word_total={sum(word_counts.values())} "
        f"unique_words={len(word_counts)} missing_txt={missing_txt} "
        f"empty_word_records={empty_word_records} elapsed_sec={time.time() - started:.1f}",
        flush=True,
    )

    output = {
        "mapping": args.mapping,
        "records": len(records),
        "full_records": len(mapping),
        "workers": args.workers,
        "word_total": sum(word_counts.values()),
        "unique_words": len(word_counts),
        "missing_txt": missing_txt,
        "empty_word_records": empty_word_records,
        "tokenizers": {},
    }

    for name, (path, processor) in processors.items():
        unk_id = processor.unk_id()
        word_total = sum(word_counts.values())
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
                    f"encode_progress tokenizer={name} {index}/{len(word_counts)} "
                    f"elapsed_sec={time.time() - started:.1f}",
                    flush=True,
                )

        oov_words = set(oov_counts)
        records_with_unk = sum(1 for words in record_words.values() if words & oov_words)
        unique_words = len(word_counts)
        unique_oov_words = len(oov_words)
        result = {
            "path": path,
            "vocab_size": processor.vocab_size(),
            "unk_id": unk_id,
            "unk_piece": processor.id_to_piece(unk_id),
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
        output["tokenizers"][name] = result

        print(f"RESULT tokenizer={name}", flush=True)
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

    output["elapsed_sec"] = time.time() - started
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote={output_path}", flush=True)
    print(f"elapsed_sec_total={output['elapsed_sec']:.1f}", flush=True)


if __name__ == "__main__":
    main()
