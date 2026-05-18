#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import Iterable

import torch
from omegaconf import OmegaConf

import lcasr
from exp.train import train
from lcasr.optim import madgrad
from lcasr.utils.augmentation import SpecAugment
from lcasr.utils.dataloading import VariableBatchSimpleDataloader
from lcasr.utils.general import get_model_class, load_checkpoint, load_model
from lcasr.utils.helpers import exists
from lcasr.utils.scheduling import ConstantLRScheduler, CosineLRScheduler, SequenceWarmupManager

try:
    from apex.optimizers import FusedAdam
except Exception:
    FusedAdam = torch.optim.Adam


def subset_pairs(pairs, max_records):
    if max_records is None:
        return pairs
    keys = sorted(pairs.keys())[:max_records]
    return {key: pairs[key] for key in keys}


def acoustic_state_from_ssl_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if "acoustic_model" in checkpoint:
        return checkpoint["acoustic_model"], "acoustic_model"
    if "model" not in checkpoint:
        raise KeyError(f"{path} has neither 'acoustic_model' nor 'model'")
    state = checkpoint["model"]
    stripped = {}
    for key, value in state.items():
        if key.startswith("model."):
            stripped[key[len("model.") :]] = value
    if stripped:
        return stripped, "model.* stripped from BEST-RQ wrapper"
    return state, "model"


def load_frozen_backbone(model: torch.nn.Module, checkpoint_path: str, load_decoder: bool):
    state, source_key = acoustic_state_from_ssl_checkpoint(checkpoint_path)
    if not load_decoder:
        state = {key: value for key, value in state.items() if not key.startswith("decoder.")}
    loaded_names = set(state.keys()) & set(model.state_dict().keys())
    if not loaded_names:
        raise RuntimeError(f"no matching model keys loaded from {checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded {len(loaded_names)} tensors from {checkpoint_path} ({source_key})")
    if missing:
        print(f"missing keys after SSL load: {len(missing)}")
        print("\n".join(f"  {key}" for key in missing[:20]))
    if unexpected:
        print(f"unexpected keys after SSL load: {len(unexpected)}")
        print("\n".join(f"  {key}" for key in unexpected[:20]))


def freeze_except(model: torch.nn.Module, trainable_prefixes: Iterable[str]):
    prefixes = tuple(trainable_prefixes)
    trainable, frozen = 0, 0
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(prefixes)
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    if trainable == 0:
        raise RuntimeError(f"no trainable parameters matched prefixes: {prefixes}")
    print(f"trainable parameters: {trainable}")
    print(f"frozen parameters: {frozen}")


def build_optimizer_and_scheduler(config, params, device_type: str):
    optim_type = config.optimizer.name
    optim_args = dict(config.optimizer.args)
    if optim_type == "adam":
        optimizer = FusedAdam(params, **optim_args) if device_type == "cuda" else torch.optim.Adam(params, **optim_args)
    elif optim_type == "adamw":
        optimizer = torch.optim.AdamW(params, **optim_args)
    elif optim_type == "madgrad":
        optimizer = madgrad.MADGRAD(params, **optim_args)
    else:
        raise NotImplementedError(f"unsupported optimizer for ROB-91 probe: {optim_type}")

    scheduler_type = config.get("scheduler", {}).get("name", config.get("scheduler", {}).get("type", "cosine"))
    if scheduler_type == "constant":
        scheduler = ConstantLRScheduler(optimizer=optimizer)
    elif scheduler_type == "cosine":
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            warmup_steps=config.scheduler.warmup_steps,
            peak_value=optim_args["lr"],
            final_value=0.0,
        )
    else:
        raise NotImplementedError(f"unsupported scheduler for ROB-91 probe: {scheduler_type}")
    return optimizer, scheduler


def main(args):
    args.config_path = args.config
    config = OmegaConf.load(args.config)
    if args.disable_wandb:
        config.wandb.use = False
    if args.max_records is not None:
        config.data.max_records = args.max_records
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    args.config = config

    os.makedirs(config.checkpointing.dir, exist_ok=True)
    tokenizer_kwargs = {"tokenizer_path": config.training.tokenizer_path} if "tokenizer_path" in config.training else {}
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)

    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(12345)
    model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
    load_frozen_backbone(
        model=model,
        checkpoint_path=config.probe.ssl_checkpoint,
        load_decoder=bool(config.probe.get("load_decoder_from_ssl", False)),
    )
    freeze_except(model, config.probe.get("trainable_prefixes", ["decoder."]))
    model.print_total_params()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        config=config,
        params=[param for param in model.parameters() if param.requires_grad],
        device_type=device.type,
    )

    sequence_scheduler = None
    if "sequence_scheduler" in config:
        sequence_scheduler = SequenceWarmupManager(
            initial_batch_size=config.training.batch_size,
            initial_sequence_length=config.audio_chunking.size,
            **config.sequence_scheduler,
        )

    seen_ids, step, epoch = [], 0, 0
    if not args.reset_step:
        seen_ids, step, epoch = load_checkpoint(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sequence_scheduler=sequence_scheduler,
            path=config.checkpointing.dir,
            device=device,
        )

    if args.reset_step:
        seen_ids, step, epoch = [], 0, 0

    random_seed = config.training.get("random_seed", 1234)
    if random_seed == "random":
        random_seed = int(time.time()) % 10000
    random.seed(random_seed)

    paired_data = lcasr.utils.audio_tools.load_json(config.data.path)
    paired_data = subset_pairs(paired_data, config.data.get("max_records", None))
    dataloader = VariableBatchSimpleDataloader(
        pairs=paired_data,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        chunk_size=config.audio_chunking.size,
        chunk_overlap=config.audio_chunking.overlap,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch=args.prefetch_factor,
        seen_ids=seen_ids,
        random_seed=random_seed,
    )

    augmentation = SpecAugment(**config.spec_augment) if "spec_augment" in config else None
    assert exists(augmentation) or config.training.get("start_spec_augment_after_n_epochs", -1) == -1

    train(
        args=args,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        sequence_scheduler=sequence_scheduler,
        device=device,
        seen_ids=seen_ids,
        step=step,
        epoch=epoch,
        augmentation=augmentation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", required=True)
    parser.add_argument("-rm_sched", "--remove_scheduler", action="store_true")
    parser.add_argument("-reset_step", "--reset_step", action="store_true")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=0)
    parser.add_argument("-pin_memory", "--pin_memory", action="store_true")
    parser.add_argument("-prefetch", "--prefetch_factor", type=int, default=1)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("-debug_hooks", "--debug_hooks", action="store_true")
    main(parser.parse_args())
