#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf

import lcasr
from exp.train import train
from lcasr.models.base import LayerNorm, RMSNorm
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


class BiLSTMCTCProbeHead(nn.Module):
    def __init__(
            self,
            d_model: int,
            vocab_size: int,
            norm: bool = False,
            norm_fn=LayerNorm,
            hidden_size: int = 1024,
            num_layers: int = 2,
            dropout: float = 0.2,
        ):
        super().__init__()
        self.num_classes = vocab_size + 1
        self.norm = norm_fn(d_model) if norm else nn.Identity()
        dropout = dropout if num_layers > 1 else 0.0
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.ff = nn.Linear(hidden_size * 2, self.num_classes)
        self.reprojection = nn.Linear(self.num_classes, d_model)

    def forward(self, x, logits=False):
        x_norm = self.norm(x)
        with torch.cuda.amp.autocast(enabled=False):
            x, _ = self.bilstm(x_norm.float())
            x = self.ff(x)
        return x if logits else F.log_softmax(x, dim=-1)

    def project_back(self, x):
        return self.reprojection(x)

    def integrate_projections(self, x, proj1):
        return x + proj1


class FrozenBackboneCTCProbe(nn.Module):
    def __init__(self, acoustic_model: nn.Module, decoder: nn.Module):
        super().__init__()
        self.acoustic_model = acoustic_model
        self.decoder = decoder

    @property
    def subsampling(self):
        return self.acoustic_model.subsampling

    def print_total_params(self, only_trainable=False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad) if only_trainable else sum(p.numel() for p in self.parameters())
        pstr = "Total trainable params: " if only_trainable else "Total params: "
        print(f"{pstr}: ", total / 1e6, "M")
        return total

    def forward(self, *args, **kwargs):
        return_logits = kwargs.get("return_logits", False)
        kwargs["skip_vocab_projection"] = True
        output = self.acoustic_model(*args, **kwargs)
        hidden_states = output["hidden_states"]
        if self.acoustic_model.legasee_double_norm:
            hidden_states = self.decoder.norm(hidden_states)
        final_posts = self.decoder(x=hidden_states, logits=return_logits)
        return {"final_posteriors": final_posts, "length": output["length"]}

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(normalize_probe_state_dict(self, state_dict), strict=strict)


def _norm_fn(config):
    return RMSNorm if config.model.get("default_norm", "layer_norm") == "rms_norm" else LayerNorm


def build_probe_model(config, vocab_size: int, model_class=None):
    acoustic_model = load_model(config, vocab_size, model_class or get_model_class(config=config))
    if config.get("probe", {}).get("head", "linear") != "bilstm":
        return acoustic_model

    head = BiLSTMCTCProbeHead(
        d_model=config.model.d_model,
        vocab_size=vocab_size,
        norm=config.model.get("decoder_norm", False),
        norm_fn=_norm_fn(config),
        hidden_size=config.probe.get("bilstm_hidden_size", 1024),
        num_layers=config.probe.get("bilstm_num_layers", 2),
        dropout=config.probe.get("bilstm_dropout", 0.2),
    )
    return FrozenBackboneCTCProbe(acoustic_model=acoustic_model, decoder=head)


def acoustic_model_from_probe(model: nn.Module):
    return model.acoustic_model if isinstance(model, FrozenBackboneCTCProbe) else model


def normalize_probe_state_dict(model: nn.Module, state_dict):
    if not isinstance(model, FrozenBackboneCTCProbe):
        return state_dict
    if any(key.startswith("acoustic_model.") for key in state_dict):
        return state_dict
    if not any(key.startswith("final_decoder.") for key in state_dict):
        return state_dict

    converted = {}
    for key, value in state_dict.items():
        if key.startswith("final_decoder."):
            converted[f"decoder.{key[len('final_decoder.'):]}"] = value
        else:
            converted[f"acoustic_model.{key}"] = value
    return converted


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


def init_wandb(config, config_path: str, total_params: int):
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("use", False):
        return None
    wandb_dir = wandb_config.get("dir", "./wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    init_kwargs = {
        "project": wandb_config["project_name"],
        "config": OmegaConf.to_container(config, resolve=True),
        "name": wandb_config.get("name", None),
        "dir": wandb_dir,
    }
    run_id = wandb_config.get("id", "")
    if run_id:
        run = wandb.init(id=run_id, resume="must", allow_val_change=True, **init_kwargs)
    else:
        run = wandb.init(**init_kwargs)
    wandb.config.update({"total_params": total_params}, allow_val_change=True)
    print(f"\nLogging with WandB id: {wandb.run.id}\n")
    config.wandb.id = wandb.run.id
    if wandb_config.get("update_config_with_wandb_id", False):
        OmegaConf.save(config=config, f=config_path)
    return run


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
    model = build_probe_model(config, tokenizer.vocab_size())
    load_frozen_backbone(
        model=acoustic_model_from_probe(model),
        checkpoint_path=config.probe.ssl_checkpoint,
        load_decoder=bool(config.probe.get("load_decoder_from_ssl", False)),
    )
    freeze_except(model, config.probe.get("trainable_prefixes", ["decoder."]))
    total_params = model.print_total_params()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = init_wandb(config, args.config_path, total_params)
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

    try:
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
    finally:
        if wandb_run is not None:
            wandb.finish()


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
