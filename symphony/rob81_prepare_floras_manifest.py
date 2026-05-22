#!/usr/bin/env python3
"""Prepare normalized Floras manifest for ROB-81 supervised finetuning."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import sentencepiece as spm

from symphony.rob81_floras_oov_audit import WORD_RE, normalize_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-output-dir", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--unk-id", type=int, default=1)
    parser.add_argument("--examples", type=int, default=20)
    return parser.parse_args()


def safe_name(record_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id).strip("_") or "record"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def iter_items(mapping: Dict[str, Dict[str, Any]], limit: int) -> Iterable[tuple[str, Dict[str, Any]]]:
    items = mapping.items()
    if limit > 0:
        items = list(items)[:limit]
    return items


def word_has_unk(
    tokenizer: spm.SentencePieceProcessor,
    word: str,
    unk_id: int,
    cache: Dict[str, bool],
) -> bool:
    if word not in cache:
        cache[word] = unk_id in tokenizer.encode(word)
    return cache[word]


def oov_words(
    tokenizer: spm.SentencePieceProcessor,
    words: Iterable[str],
    unk_id: int,
    cache: Dict[str, bool],
) -> list[str]:
    return [
        word
        for word in sorted(set(words))
        if word_has_unk(tokenizer, word, unk_id, cache)
    ]


def normalize_timestamps(timestamps: list[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[str]]:
    normalized_timestamps = []
    normalized_words = []
    for element in timestamps:
        normalized_text = normalize_safe(str(element.get("text", "")))
        if not normalized_text:
            continue
        normalized_element = dict(element)
        normalized_element["text"] = normalized_text
        normalized_timestamps.append(normalized_element)
        normalized_words.extend(WORD_RE.findall(normalized_text))
    return normalized_timestamps, normalized_words


def main() -> None:
    args = parse_args()
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    mapping = load_json(args.mapping)
    output_dir = Path(args.text_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)

    filtered_mapping = {}
    stats = {
        "input_records": 0,
        "kept_records": 0,
        "dropped_missing_text": 0,
        "dropped_empty_after_normalization": 0,
        "dropped_oov_after_normalization": 0,
        "unique_words_checked": 0,
        "examples": [],
    }
    unk_cache: Dict[str, bool] = {}

    for record_id, record in iter_items(mapping, args.limit):
        stats["input_records"] += 1
        txt_path = str(record.get("txt", ""))
        if not txt_path or not os.path.exists(txt_path):
            stats["dropped_missing_text"] += 1
            continue

        payload = load_json(txt_path)
        timestamps = payload.get("word_timestamps", [])
        normalized_timestamps, normalized_words = normalize_timestamps(timestamps)
        if not normalized_words:
            stats["dropped_empty_after_normalization"] += 1
            continue

        record_oov_words = oov_words(tokenizer, normalized_words, args.unk_id, unk_cache)
        if record_oov_words:
            stats["dropped_oov_after_normalization"] += 1
            if len(stats["examples"]) < args.examples:
                stats["examples"].append(
                    {"record_id": record_id, "txt": txt_path, "oov_words": record_oov_words[:20]}
                )
            continue

        normalized_payload = dict(payload)
        normalized_payload["word_timestamps"] = normalized_timestamps
        normalized_txt_path = output_dir / f"{safe_name(record_id)}.json"
        write_json(str(normalized_txt_path), normalized_payload)

        filtered_record = dict(record)
        filtered_record["txt"] = str(normalized_txt_path)
        filtered_mapping[record_id] = filtered_record
        stats["kept_records"] += 1

    stats["unique_words_checked"] = len(unk_cache)
    write_json(args.output, filtered_mapping)
    write_json(args.summary_json, stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
