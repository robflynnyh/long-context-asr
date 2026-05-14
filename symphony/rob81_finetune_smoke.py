#!/usr/bin/env python3
"""CPU smoke check for ROB-81 supervised encoder-decoder Floras finetuning."""

from __future__ import annotations

import argparse
from itertools import islice

import torch
from omegaconf.omegaconf import OmegaConf

import lcasr
from exp.train_files.train_enc_dec import load_pretrained_model_state
from lcasr.utils.dataloading import VariableBatchSimpleDataloader, chunk_text_json
from lcasr.utils.general import get_model_class, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-records", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if "flash_attn" in config.model:
        config.model.flash_attn = False

    tokenizer_path = config.training.get("tokenizer_path", None)
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**({"tokenizer_path": tokenizer_path} if tokenizer_path else {}))

    model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
    load_pretrained_model_state(
        model=model,
        path=config.checkpointing.pretrained,
        device=torch.device("cpu"),
    )
    print(f"loaded model params={sum(param.numel() for param in model.parameters())}")

    paired_data = lcasr.utils.audio_tools.load_json(config.data.path)
    if len(paired_data) == 0:
        raise RuntimeError(f"filtered manifest is empty: {config.data.path}")
    small_data = dict(islice(paired_data.items(), args.max_records))
    dataloader = VariableBatchSimpleDataloader(
        pairs=small_data,
        tokenizer=tokenizer,
        batch_size=min(config.training.batch_size, max(1, len(small_data))),
        chunk_size=config.audio_chunking.size,
        chunk_overlap=config.audio_chunking.overlap,
        num_workers=0,
        pin_memory=False,
        prefetch=None,
        random_seed=config.training.get("random_seed", 1234),
    )

    audio, audio_lengths, text, ids = next(iter(dataloader))
    text_chunks = [
        chunk_text_json(
            text=entry,
            chunk_size=config.audio_chunking.size,
            chunk_overlap=config.audio_chunking.overlap,
            spectogram_length=int(audio.shape[-1]),
        )
        for entry in text
    ]
    encoded_nonempty = [
        tokenizer.encode(chunk)
        for chunks in text_chunks
        for chunk in chunks
        if chunk.strip()
    ]
    if not encoded_nonempty:
        raise RuntimeError("smoke batch had no nonempty encoded text chunks")

    print(
        "smoke_ok "
        f"records={len(paired_data)} batch={len(ids)} "
        f"audio_shape={tuple(audio.shape)} max_audio_len={int(audio_lengths.max())} "
        f"encoded_chunks={len(encoded_nonempty)}"
    )


if __name__ == "__main__":
    main()
