import argparse
import os
import random
import time
from typing import Any, List, Union

import lcasr
import torch
import wandb
from contextlib import nullcontext
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from lcasr.utils.dataloading import VariableBatchSimpleDataloader, reset_seen_ids
from lcasr.utils.general import get_model_class, load_checkpoint, load_model, load_optimizer, save_model
from lcasr.utils.streaming_targets import (
    build_streaming_frame_targets,
    filter_words_by_end_frame,
    pad_audio_for_streaming_delay,
)


class SyntheticStreamingDataloader:
    def __init__(self, tokenizer: Any, batch_size: int = 1, num_batches: int = 1, frames: int = 96, feat_in: int = 80):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.frames = frames
        self.feat_in = feat_in

    def __len__(self):
        return self.num_batches

    def total_recordings(self):
        return self.num_batches * self.batch_size

    def update(self, batch_size: int, seen_ids: List[str] = [], random_seed: Union[int, str] = "same"):
        self.batch_size = batch_size

    def __iter__(self):
        for batch_idx in range(self.num_batches):
            audio = torch.randn(self.batch_size, self.feat_in, self.frames)
            audio_lengths = torch.full((self.batch_size,), self.frames, dtype=torch.long)
            text = []
            ids = []
            for item_idx in range(self.batch_size):
                text.append(
                    [
                        {"startTime": "0.10s", "endTime": "0.30s", "word": "hello"},
                        {"startTime": "0.55s", "endTime": "0.75s", "word": "world"},
                    ]
                )
                ids.append(f"synthetic-{batch_idx}-{item_idx}")
            yield audio, audio_lengths, text, ids


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"invalid dtype: {dtype}")


def make_dataloader(config, tokenizer, args, seen_ids):
    if config["data"].get("synthetic", False):
        synthetic = config["data"].get("synthetic_args", {})
        return SyntheticStreamingDataloader(
            tokenizer=tokenizer,
            batch_size=config["training"]["batch_size"],
            num_batches=synthetic.get("num_batches", 1),
            frames=synthetic.get("frames", config["audio_chunking"]["size"]),
            feat_in=config["model"].get("feat_in", 80),
        )

    paired_data = lcasr.utils.audio_tools.load_json(config["data"]["path"])
    max_records = config["data"].get("max_records", None)
    if max_records is not None:
        paired_data = dict(list(paired_data.items())[: int(max_records)])

    return VariableBatchSimpleDataloader(
        pairs=paired_data,
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        chunk_size=config.audio_chunking["size"],
        chunk_overlap=config.audio_chunking["overlap"],
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch=args.prefetch_factor,
        seen_ids=seen_ids,
        random_seed=config["training"].get("random_seed", 1234),
    )


def train(args, model, dataloader, optimizer, scheduler, device, step=0, seen_ids=None, epoch=0):
    seen_ids = [] if seen_ids is None else seen_ids
    scaler = GradScaler(enabled=torch.cuda.is_available())
    dtype = get_dtype(args.config["training"].get("dtype", "bfloat16"))
    clip_value = args.config["training"].get("clip_value", 0.8)
    max_epochs = args.config["training"].get("max_epochs", 1)
    max_steps = args.config["training"].get("max_steps", float("inf"))
    scheduler_total_steps = args.config["training"].get(
        "scheduler_total_steps",
        max_steps if max_steps != float("inf") else dataloader.total_recordings() * max_epochs,
    )
    backprop_every = args.config["training"].get("backprop_every", 1)
    delay_seconds = args.config["streaming"].get("delay_seconds", 2.0)
    buffer_seconds = args.config["streaming"].get("buffer_seconds", 0.25)
    chunk_size = args.config["audio_chunking"]["size"]
    chunk_overlap = args.config["audio_chunking"].get("overlap", 0)
    assert chunk_size > chunk_overlap, "audio_chunking.size must be greater than overlap"
    model_dtype = next(model.parameters()).dtype
    optimizer.zero_grad()
    global_step = step

    for cur_epoch in range(epoch, max_epochs):
        pbar = tqdm(dataloader, desc=f"Streaming decoder training - Epoch {cur_epoch}")
        for batch in pbar:
            audio, audio_lengths, transcripts, ids = batch
            seen_ids.extend(ids)
            stride = chunk_size - chunk_overlap
            for chunk_start in range(0, int(audio_lengths.max().item()), stride):
                active = audio_lengths > chunk_start
                if active.sum().item() == 0:
                    continue
                chunk = audio[active, :, chunk_start : chunk_start + chunk_size]
                chunk_lengths = torch.clamp(audio_lengths[active] - chunk_start, min=0, max=chunk.size(-1))
                chunk_transcripts = [
                    filter_words_by_end_frame(transcripts[i], chunk_start, chunk_start + chunk_size)
                    for i, keep in enumerate(active.tolist())
                    if keep
                ]
                chunk_starts = torch.full((active.sum().item(),), chunk_start, dtype=torch.long)
                chunk, chunk_lengths = pad_audio_for_streaming_delay(
                    chunk,
                    chunk_lengths,
                    delay_seconds=delay_seconds,
                    buffer_seconds=buffer_seconds,
                )
                chunk = chunk.to(device=device, dtype=model_dtype)
                chunk_lengths = chunk_lengths.to(device)
                output_lengths = model.output_lengths(chunk_lengths)
                frame_targets = build_streaming_frame_targets(
                    transcripts=chunk_transcripts,
                    output_lengths=output_lengths.cpu(),
                    tokenizer=dataloader.tokenizer,
                    subsampling_factor=model.subsampling_factor,
                    delay_seconds=delay_seconds,
                    chunk_start_frames=chunk_starts,
                    silence_id=model.get_silence_id(),
                ).to(device)

                with torch.autocast(device.type, dtype=dtype) if torch.cuda.is_available() else nullcontext():
                    out = model.calc_loss(audio_signal=chunk, length=chunk_lengths, frame_targets=frame_targets)
                    loss = out["loss"] / backprop_every

                scaler.scale(loss).backward()
                if (global_step + 1) % backprop_every == 0:
                    scaler.unscale_(optimizer)
                    if clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler.is_warmup:
                        scheduler.step()
                        if not scheduler.is_warming_up():
                            scheduler.set_cosine_schedule(total_recordings=scheduler_total_steps, cur_podcast=global_step)
                    else:
                        scheduler.step(epoch=global_step)

                global_step += 1
                if args.config["wandb"].get("use", False):
                    wandb.log(
                        {
                            **out["display_losses"],
                            "learning_rate": scheduler.get_last_lr()[0],
                            "step": global_step,
                        }
                    )
                pbar.set_postfix({"loss": f"{out['display_losses']['loss']:.4f}", "step": global_step})
                if global_step >= max_steps:
                    save_model(model, optimizer, scheduler, global_step, args.config, seen_ids=seen_ids, epoch=cur_epoch)
                    return model, global_step, cur_epoch
        seen_ids = reset_seen_ids(seen_ids, epoch=cur_epoch)

    save_model(model, optimizer, scheduler, global_step, args.config, seen_ids=seen_ids, epoch=max_epochs)
    return model, global_step, max_epochs


def main(args):
    args.config_path = args.config
    args.config = OmegaConf.load(args.config)
    if args.checkpoint_dir is not None:
        args.config["checkpointing"]["dir"] = args.checkpoint_dir
    if args.batch_size is not None:
        args.config["training"]["batch_size"] = args.batch_size
    if args.data_path is not None:
        args.config["data"]["path"] = args.data_path
    if args.max_records is not None:
        args.config["data"]["max_records"] = args.max_records
    if args.max_steps is not None:
        args.config["training"]["max_steps"] = args.max_steps
    if args.disable_wandb:
        args.config["wandb"]["use"] = False
    os.makedirs(args.config["checkpointing"]["dir"], exist_ok=True)

    tokenizer_kwargs = {}
    if "tokenizer_path" in args.config["training"]:
        tokenizer_kwargs["tokenizer_path"] = args.config["training"]["tokenizer_path"]
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)

    seed = args.config["training"].get("random_seed", 1234)
    if seed == "random":
        seed = int(time.time()) % 10000
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = load_model(args.config, tokenizer.vocab_size(), get_model_class(config=args.config))
    total_params = model.print_total_params()
    if args.config["wandb"].get("use", False):
        wandb_config = args.config["wandb"]
        wandb.init(
            project=wandb_config["project_name"],
            name=wandb_config.get("name", None),
            id=wandb_config.get("id", "") or None,
            resume="must" if wandb_config.get("id", "") else None,
            dir=wandb_config.get("dir", "./wandb"),
            config=OmegaConf.to_container(args.config, resolve=True),
            allow_val_change=True,
        )
        wandb.config.update({"total_params": total_params}, allow_val_change=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer, scheduler = load_optimizer(args.config, model)

    seen_ids, step, epoch = load_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        path=args.config["checkpointing"]["dir"],
        device=device,
    )
    if args.reset_step:
        seen_ids, step, epoch = [], 0, 0

    dataloader = make_dataloader(args.config, tokenizer, args, seen_ids)
    print(f"Streaming decoder ASR params: {total_params / 1e6:.2f}M")
    print(f"Starting from step: {step}")
    train(args, model, dataloader, optimizer, scheduler, device, step=step, seen_ids=seen_ids, epoch=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", type=str, required=True)
    parser.add_argument("-rm_sched", "--remove_scheduler", action="store_true")
    parser.add_argument("-reset_step", "--reset_step", action="store_true")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=0)
    parser.add_argument("-pin_memory", "--pin_memory", action="store_true")
    parser.add_argument("-prefetch", "--prefetch_factor", type=int, default=1)
    parser.add_argument("-checkpoint_dir", "--checkpoint_dir", type=str, default=None)
    parser.add_argument("-batch_size", "--batch_size", type=int, default=None)
    parser.add_argument("-data_path", "--data_path", type=str, default=None)
    parser.add_argument("-max_records", "--max_records", type=int, default=None)
    parser.add_argument("-max_steps", "--max_steps", type=int, default=None)
    parser.add_argument("-disable_wandb", "--disable_wandb", action="store_true")
    main(parser.parse_args())
