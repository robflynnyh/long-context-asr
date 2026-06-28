#!/usr/bin/env python3
"""Bounded ROB-320 CPU smoke using a real Spotify manifest sample."""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

import lcasr.utils.audio_tools
from lcasr.models.sconformer_xl import SCConformerXL
from lcasr.utils.dataloading import SimpleDataset
from lcasr.utils.general import load_checkpoint, save_model
from lcasr.utils.helpers import load_json


def log(message: str):
    print(message, flush=True)


def load_short_sample(manifest_path: str, max_frames: int, max_candidates: int):
    log(f"loading manifest: {manifest_path}")
    pairs = load_json(manifest_path)
    log(f"manifest records: {len(pairs)}")
    candidates = list(pairs.items())[:max_candidates]
    candidates = sorted(candidates, key=lambda item: float(item[1].get("duration", 0.0)))
    for sample_id, entry in candidates:
        if not os.path.exists(entry["audio"]) or not os.path.exists(entry["txt"]):
            continue
        log(f"loading sample candidate: {sample_id}")
        audio = torch.load(entry["audio"], map_location="cpu")
        transcript = load_json(entry["txt"])
        words = SimpleDataset.resolve_txt(transcript)
        text = " ".join(word.get("word", word.get("text", "")) for word in words).strip()
        if not text:
            continue
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        if audio.dim() != 3:
            continue
        frames = min(int(audio.shape[-1]), max_frames)
        if frames < 64:
            continue
        log(f"selected sample: {sample_id} frames={frames}")
        return sample_id, audio[..., :frames].float(), text, len(pairs)
    raise RuntimeError(f"no usable sample found in first {max_candidates} entries of {manifest_path}")


def build_smoke_config(vocab_size: int, checkpoint_dir: str, model_size: str):
    if model_size == "tiny":
        model = {
            "vocab_size": vocab_size,
            "feat_in": 80,
            "subsampling": "dw_striding",
            "subsampling_factor": 4,
            "subsampling_conv_channels": 32,
            "subsampling_act": "silu",
            "subsampling_norm_out": False,
            "n_layers": 1,
            "d_model": 64,
            "n_heads": 2,
            "head_dim": 32,
            "dropout_ff": 0.0,
            "dropout_conv": 0.0,
            "dropout_attn": 0.0,
            "conv_kernel_size": 9,
            "conv_expansion_factor": 1,
            "decoder_norm": True,
            "use_rotary": True,
            "rotary_base_freq": 1500000,
            "self_conditioning": False,
            "default_norm": "layer_norm",
            "bias_in_ff": False,
            "gru_module": True,
            "gru_hidden_size": None,
            "gru_num_layers": 1,
            "gru_dropout": 0.0,
            "gru_bidirectional": False,
        }
    elif model_size == "final":
        model = {
            "vocab_size": vocab_size,
            "feat_in": 80,
            "subsampling": "dw_striding",
            "subsampling_factor": 8,
            "subsampling_conv_channels": 256,
            "subsampling_act": "silu",
            "subsampling_norm_out": False,
            "n_layers": 18,
            "d_model": 1024,
            "n_heads": 8,
            "head_dim": 128,
            "dropout_ff": 0.0,
            "dropout_conv": 0.0,
            "dropout_attn": 0.0,
            "conv_kernel_size": 9,
            "conv_expansion_factor": 1,
            "decoder_norm": True,
            "use_rotary": True,
            "rotary_base_freq": 1500000,
            "self_conditioning": True,
            "default_norm": "layer_norm",
            "bias_in_ff": False,
            "checkpoint_every_n_layers": 1,
            "ff_checkpoint_lvl": 2,
            "gru_module": True,
            "gru_hidden_size": None,
            "gru_num_layers": 1,
            "gru_dropout": 0.0,
            "gru_bidirectional": False,
        }
    else:
        raise ValueError(f"unknown model size: {model_size}")

    return OmegaConf.create({
        "model": {
            **model,
        },
        "checkpointing": {"dir": checkpoint_dir},
    })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json")
    parser.add_argument("--artifact-root", default="/mnt/parscratch/users/acp21rjf/symphony-job-artifacts/ROB-320")
    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=5000)
    parser.add_argument("--model-size", choices=("tiny", "final"), default="tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    smoke_id = os.environ.get("SLURM_JOB_ID", "local")
    smoke_prefix = f"{args.device}_{args.model_size}_smoke"
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.artifact_root, "checkpoints", f"{smoke_prefix}_{smoke_id}")
    summary_json = args.summary_json or os.path.join(args.artifact_root, smoke_prefix, f"summary_{smoke_id}.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_json), exist_ok=True)

    log("loading tokenizer")
    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    sample_id, audio, text, manifest_count = load_short_sample(args.manifest, args.max_frames, args.max_candidates)
    config = build_smoke_config(tokenizer.vocab_size(), checkpoint_dir, args.model_size)

    log("building GRU smoke model")
    model = SCConformerXL(**config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    log("running forward/backward")
    audio = audio.to(device)
    lengths = torch.LongTensor([audio.shape[-1]]).to(device)
    out = model(audio_signal=audio, length=lengths)
    output_length = int(out["length"][0].item())
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise RuntimeError(f"selected sample has no tokenizer output: {sample_id}")
    target_length = max(1, min(len(token_ids), max(1, output_length // 2)))
    targets = torch.LongTensor(token_ids[:target_length]).unsqueeze(0).to(device)
    target_lengths = torch.LongTensor([target_length]).to(device)

    ctc_loss = torch.nn.CTCLoss(blank=model.decoder.num_classes - 1, reduction="mean", zero_infinity=True)
    loss = ctc_loss(out["final_posteriors"].transpose(0, 1), targets, out["length"], target_lengths)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke loss: {loss}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    log("saving checkpoint")
    save_model(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        podcast_step=1,
        config=config,
        sequence_scheduler=None,
        seen_ids=[sample_id],
        epoch=0,
    )

    log("loading checkpoint")
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loaded_model = SCConformerXL(**config.model)
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-4)
    seen_ids, step, epoch = load_checkpoint(
        args=SimpleNamespace(remove_scheduler=False),
        model=loaded_model,
        optimizer=loaded_optimizer,
        scheduler=None,
        sequence_scheduler=None,
        path=checkpoint_dir,
        device="cpu",
    )
    if step != 1 or seen_ids != [sample_id] or epoch != 0:
        raise RuntimeError(f"unexpected checkpoint state: step={step}, seen_ids={seen_ids}, epoch={epoch}")

    summary = {
        "status": "success",
        "manifest": args.manifest,
        "manifest_records": manifest_count,
        "sample_id": sample_id,
        "frames": int(audio.shape[-1]),
        "output_length": output_length,
        "target_length": target_length,
        "loss": float(loss.item()),
        "device": args.device,
        "model_size": args.model_size,
        "checkpoint_dir": checkpoint_dir,
        "summary_json": summary_json,
    }
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
