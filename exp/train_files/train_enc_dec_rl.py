import argparse
import os
import random
import resource
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import lcasr
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf.omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm

from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.dataloading import (
    VariableBatchSimpleDataloader,
    chunk_spectogram,
    chunk_text_json,
    load_sample,
    reset_seen_ids,
)
from lcasr.utils.general import (
    find_latest_checkpoint,
    get_model_class,
    load_model,
    load_optimizer,
    save_model,
)

try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    EnglishTextNormalizer = None


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"invalid dtype: {dtype}")


def normalize_text(text: str, normalizer: Any = None) -> str:
    if normalizer is not None:
        text = normalizer(text)
    return " ".join(text.lower().strip().split())


def resolve_checkpoint_path(path: str) -> str:
    if os.path.isdir(path):
        latest = find_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"no .pt checkpoints found in {path}")
        return os.path.join(path, latest)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def load_checkpoint_config(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if "config" not in checkpoint:
        raise KeyError(f"checkpoint {path} has no config key")
    return checkpoint


def merge_model_config(config: OmegaConf, checkpoint: Dict[str, Any]) -> OmegaConf:
    checkpoint_config = OmegaConf.create(checkpoint["config"])
    config.model = checkpoint_config.model
    config.model_class = checkpoint_config.get("model_class", config.get("model_class", "EncDecSconformerV2"))
    return config


def remap_legacy_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.endswith(".norm.scale"):
            new_key = new_key[: -len(".scale")] + ".weight"
        elif new_key.endswith(".out_proj.0.scale"):
            new_key = new_key[: -len(".scale")] + ".weight"
        remapped[new_key] = value
    return remapped


def load_model_state_compat(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
    return model.load_state_dict(remap_legacy_state_dict_keys(state_dict), strict=False)


def load_or_initialize_model(config: OmegaConf, tokenizer: Any, device: torch.device, reset_step: bool):
    pretrained_path = config.checkpointing.get("pretrained", None)
    checkpoint = None
    checkpoint_path = None

    if pretrained_path is not None:
        checkpoint_path = resolve_checkpoint_path(pretrained_path)
        checkpoint = load_checkpoint_config(checkpoint_path)
        config = merge_model_config(config, checkpoint)

    if device.type == "cpu":
        config.model.flash_attn = False

    model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
    model.to(device)
    optimizer, scheduler = load_optimizer(config, model)

    seen_ids, step, epoch = [], 0, 0
    resume_path = None if reset_step else find_latest_checkpoint(config.checkpointing.dir)
    if resume_path is not None:
        resume_path = os.path.join(config.checkpointing.dir, resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_path = resume_path

    if checkpoint is not None:
        load_model_state_compat(model, checkpoint["model"])
        if not reset_step and checkpoint_path == resume_path:
            if checkpoint.get("optimizer", None) is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint.get("scheduler", None) is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            seen_ids = checkpoint.get("seen_ids", [])
            step = checkpoint.get("podcast_step", 0)
            epoch = checkpoint.get("epoch", 0)
        print(f"Loaded model from {checkpoint_path}")

    return config, model, optimizer, scheduler, seen_ids, step, epoch


def scheduler_step(scheduler: torch.optim.lr_scheduler._LRScheduler, step: int, max_steps: int) -> None:
    was_warmup = scheduler.is_warmup
    if was_warmup:
        scheduler.is_warmup = scheduler.is_warming_up()
        if not scheduler.is_warmup and was_warmup:
            scheduler.set_cosine_schedule(total_recordings=max_steps, cur_podcast=step)

    if scheduler.is_warmup:
        scheduler.step()
    else:
        scheduler.step(epoch=step)


def decode_actions(
    actions: Sequence[int],
    tokenizer: Any,
    eos_id: int,
    pad_id: int,
    normalizer: Any = None,
) -> str:
    filtered = [token for token in actions if token not in {eos_id, pad_id}]
    if len(filtered) == 0:
        return ""
    return normalize_text(tokenizer.decode(filtered), normalizer)


def perfect_wer_rewards(hypotheses: List[str], references: List[str]) -> torch.Tensor:
    rewards = []
    for hyp, ref in zip(hypotheses, references):
        wer, *_ = word_error_rate_detail(hypotheses=[hyp], references=[ref])
        rewards.append(1.0 if wer == 0.0 else 0.0)
    return torch.tensor(rewards, dtype=torch.float32)


def weighted_error_rewards(
    hypotheses: List[str],
    references: List[str],
    wer_weight: float = 0.7,
    cer_weight: float = 0.3,
    reward_offset: float = 1.0,
    reward_scale: float = 1.0,
    reward_min: Optional[float] = 0.0,
    reward_max: Optional[float] = None,
    reward_positive_threshold: Optional[float] = None,
) -> torch.Tensor:
    rewards = []
    weight_sum = wer_weight + cer_weight
    if weight_sum <= 0:
        raise ValueError("reward weights must sum to a positive value")
    wer_weight = wer_weight / weight_sum
    cer_weight = cer_weight / weight_sum

    for hyp, ref in zip(hypotheses, references):
        wer, *_ = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)
        cer, *_ = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=True)
        error = wer_weight * float(wer) + cer_weight * float(cer)
        reward = reward_offset - reward_scale * error
        if reward_min is not None:
            reward = max(float(reward_min), reward)
        if reward_max is not None:
            reward = min(float(reward_max), reward)
        if reward_positive_threshold is not None and reward <= reward_positive_threshold:
            reward = 0.0
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)


def _optional_float(config: Any, key: str, default: Optional[float]) -> Optional[float]:
    value = config.get(key, default)
    if value is None:
        return None
    return float(value)


def compute_rewards(
    hypotheses: List[str],
    references: List[str],
    algorithm: str,
    reward_config: Any = None,
) -> torch.Tensor:
    if algorithm == "grpo":
        reward_config = reward_config or {}
        reward_type = reward_config.get("reward_type", "weighted_error")
        if reward_type != "weighted_error":
            raise ValueError(f"unknown GRPO reward_type {reward_type}")
        return weighted_error_rewards(
            hypotheses=hypotheses,
            references=references,
            wer_weight=float(reward_config.get("reward_wer_weight", 0.7)),
            cer_weight=float(reward_config.get("reward_cer_weight", 0.3)),
            reward_offset=float(reward_config.get("reward_offset", 1.0)),
            reward_scale=float(reward_config.get("reward_scale", 1.0)),
            reward_min=_optional_float(reward_config, "reward_min", 0.0),
            reward_max=_optional_float(reward_config, "reward_max", None),
            reward_positive_threshold=_optional_float(reward_config, "reward_positive_threshold", None),
        )
    return perfect_wer_rewards(hypotheses=hypotheses, references=references)


def compute_advantages(
    rewards: torch.Tensor,
    group_size: int,
    algorithm: str,
    eps: float,
    min_group_std: float = 0.0,
) -> torch.Tensor:
    grouped = rearrange(rewards, "(b g) -> b g", g=group_size)
    mean = grouped.mean(dim=1, keepdim=True)

    if algorithm == "max_rl":
        advantages = (grouped - mean) / mean.clamp_min(eps)
        advantages = torch.where(mean > 0, advantages, torch.zeros_like(advantages))
    elif algorithm == "grpo":
        std = grouped.std(dim=1, keepdim=True, unbiased=False)
        active = std > max(eps, min_group_std)
        advantages = (grouped - mean) / std.clamp_min(eps)
        advantages = torch.where(active, advantages, torch.zeros_like(advantages))
    else:
        raise ValueError(f"unknown RL algorithm {algorithm}")

    return rearrange(advantages, "b g -> (b g)")


def resolve_max_generate(value: Union[int, str], target_lengths: torch.Tensor, encoder_lengths: torch.Tensor, buffer: int) -> int:
    if isinstance(value, int):
        return value
    if value == "target_length":
        return int(target_lengths.max().item()) + buffer
    if value == "encoder_states":
        return int(encoder_lengths.max().item())
    raise ValueError(f"unknown max_generate mode {value}")


@torch.no_grad()
def sample_rollouts(
    model: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    max_generate: int,
    num_rollouts: int,
    temperature: float,
    bos_id: int,
    eos_id: int,
) -> Tuple[List[List[int]], torch.Tensor]:
    was_training = model.training
    model.eval()

    encoder_out = model(audio_signal=audio, length=audio_lengths)
    a_hidden = encoder_out["a_hidden"].repeat_interleave(num_rollouts, dim=0)
    a_lengths = encoder_out["length"].repeat_interleave(num_rollouts, dim=0)

    total = a_hidden.shape[0]
    active_orig = list(range(total))
    cur_input = torch.full((total, 1), bos_id, dtype=torch.long, device=audio.device)
    active_hidden = a_hidden
    active_lengths = a_lengths
    cache = None
    actions = [[] for _ in range(total)]

    for _ in range(max_generate):
        decoder_out = model.language_model_decoder(
            tokens=cur_input,
            a_hidden=active_hidden,
            a_lengths=active_lengths,
            cache=cache,
            text_lengths=torch.full(
                (cur_input.shape[0],),
                cur_input.shape[1],
                dtype=torch.long,
                device=audio.device,
            ),
        )
        logits = decoder_out["logits"][:, -1, :]
        cache = decoder_out["kv_cache"]
        probs = (logits / temperature).softmax(dim=-1)
        pred = probs.multinomial(num_samples=1).squeeze(-1)

        keep_idx = []
        for row_idx, orig_idx in enumerate(active_orig):
            token = int(pred[row_idx].item())
            actions[orig_idx].append(token)
            if token != eos_id:
                keep_idx.append(row_idx)

        if len(keep_idx) == 0:
            break

        keep_idx_t = torch.tensor(keep_idx, dtype=torch.long, device=audio.device)
        active_orig = [active_orig[row_idx] for row_idx in keep_idx]
        active_hidden = active_hidden.index_select(0, keep_idx_t)
        active_lengths = active_lengths.index_select(0, keep_idx_t)
        if cache is not None:
            cache = {
                "cache": cache["cache"].index_select(2, keep_idx_t),
                "cache_lengths": cache["cache_lengths"].index_select(0, keep_idx_t),
            }
        cur_input = pred.index_select(0, keep_idx_t).unsqueeze(1)

    if was_training:
        model.train()

    return actions, encoder_out["length"]


def actions_to_tensors(
    actions: List[List[int]],
    bos_id: int,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([max(len(action), 1) for action in actions], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())

    inputs = torch.full((len(actions), max_len), pad_id, dtype=torch.long, device=device)
    targets = torch.full((len(actions), max_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(actions), max_len), dtype=torch.bool, device=device)

    for idx, action in enumerate(actions):
        if len(action) == 0:
            action = [pad_id]
        target = torch.tensor(action, dtype=torch.long, device=device)
        prompt = torch.tensor([bos_id] + action[:-1], dtype=torch.long, device=device)
        inputs[idx, : len(action)] = prompt
        targets[idx, : len(action)] = target
        mask[idx, : len(action)] = True

    return inputs, targets, mask, lengths


def sequence_logprobs(
    model: torch.nn.Module,
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    actions: List[List[int]],
    num_rollouts: int,
    bos_id: int,
    pad_id: int,
) -> torch.Tensor:
    inputs, targets, mask, lengths = actions_to_tensors(actions, bos_id=bos_id, pad_id=pad_id, device=audio.device)
    encoder_out = model(audio_signal=audio, length=audio_lengths)
    a_hidden = encoder_out["a_hidden"].repeat_interleave(num_rollouts, dim=0)
    a_lengths = encoder_out["length"].repeat_interleave(num_rollouts, dim=0)

    decoder_out = model.language_model_decoder(
        tokens=inputs,
        a_hidden=a_hidden,
        a_lengths=a_lengths,
        text_lengths=lengths,
    )
    log_probs = decoder_out["logits"].log_softmax(dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.masked_fill(~mask, 0.0).sum(dim=1)


def make_rl_chunks(
    audio: torch.Tensor,
    audio_lengths: torch.Tensor,
    txt: Sequence[Any],
    tokenizer: Any,
    chunk_size: int,
    chunk_overlap: int,
    pad_id: int,
    normalizer: Any,
) -> List[Dict[str, Any]]:
    audio_chunks = chunk_spectogram(spec=audio, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    txt_chunks = [
        chunk_text_json(text=entry, chunk_size=chunk_size, chunk_overlap=chunk_overlap, spectogram_length=audio.shape[-1])
        for entry in txt
    ]
    chunks = []
    culm_lengths_audio = torch.zeros_like(audio_lengths)

    for ix, cur_audio in enumerate(audio_chunks):
        remove_mask = culm_lengths_audio < audio_lengths
        cur_audio = cur_audio[remove_mask]
        cur_culm_lengths = culm_lengths_audio[remove_mask]
        cur_lengths = cur_audio.shape[-1] - (
            cur_culm_lengths + cur_audio.shape[-1] - audio_lengths[remove_mask] - chunk_overlap
        ).clamp(0)

        references = [normalize_text(entry[ix], normalizer) for i, entry in enumerate(txt_chunks) if bool(remove_mask[i].item())]
        tokenized = [tokenizer.encode(reference) for reference in references]
        token_lengths = torch.LongTensor([len(tokens) for tokens in tokenized])
        if len(token_lengths) == 0 or token_lengths.max() == 0:
            culm_lengths_audio[remove_mask] += cur_audio.shape[-1] - (chunk_overlap if ix != 0 else 0)
            continue

        text = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(tokens) for tokens in tokenized],
            batch_first=True,
            padding_value=pad_id,
        )
        chunks.append(
            {
                "audio": cur_audio,
                "audio_lengths": cur_lengths,
                "references": references,
                "text": text,
                "text_lengths": token_lengths,
            }
        )
        culm_lengths_audio[remove_mask] += cur_audio.shape[-1] - (chunk_overlap if ix != 0 else 0)

    return chunks


def rl_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    chunk: Dict[str, Any],
    tokenizer: Any,
    config: OmegaConf,
    device: torch.device,
    dtype: torch.dtype,
    normalizer: Any,
) -> Dict[str, Any]:
    rl_config = config.rl
    bos_id = model.get_bos_id()
    eos_id = model.get_eos_id()
    pad_id = model.get_pad_id()
    num_rollouts = int(rl_config.num_rollouts)

    model_dtype = next(model.parameters()).dtype
    audio = chunk["audio"].to(device, dtype=model_dtype)
    audio_lengths = chunk["audio_lengths"].to(device)
    target_lengths = chunk["text_lengths"].to(device)

    with autocast(device.type, dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext():
        with torch.no_grad():
            encoder_out = model(audio_signal=audio, length=audio_lengths)
            max_generate = resolve_max_generate(
                rl_config.get("max_generate", "target_length"),
                target_lengths=target_lengths,
                encoder_lengths=encoder_out["length"],
                buffer=int(rl_config.get("max_generate_buffer", 8)),
            )
        actions, _ = sample_rollouts(
            model=model,
            audio=audio,
            audio_lengths=audio_lengths,
            max_generate=max_generate,
            num_rollouts=num_rollouts,
            temperature=float(rl_config.temperature),
            bos_id=bos_id,
            eos_id=eos_id,
        )
        hypotheses = [
            decode_actions(action, tokenizer=tokenizer, eos_id=eos_id, pad_id=pad_id, normalizer=normalizer)
            for action in actions
        ]
        references = [reference for reference in chunk["references"] for _ in range(num_rollouts)]
        rewards = compute_rewards(
            hypotheses=hypotheses,
            references=references,
            algorithm=rl_config.algorithm,
            reward_config=rl_config,
        ).to(device)
        advantages = compute_advantages(
            rewards=rewards,
            group_size=num_rollouts,
            algorithm=rl_config.algorithm,
            eps=float(rl_config.get("advantage_eps", 1e-6)),
            min_group_std=float(rl_config.get("reward_std_min", 0.0)),
        ).to(device)

        if advantages.abs().sum() == 0:
            loss = torch.zeros((), device=device, requires_grad=True)
            skipped_zero_advantage = True
        else:
            was_training = model.training
            model.eval()
            try:
                logprobs = sequence_logprobs(
                    model=model,
                    audio=audio,
                    audio_lengths=audio_lengths,
                    actions=actions,
                    num_rollouts=num_rollouts,
                    bos_id=bos_id,
                    pad_id=pad_id,
                )
            finally:
                if was_training:
                    model.train()
            loss = -(advantages.detach() * logprobs).mean()
            skipped_zero_advantage = False

    optimizer.zero_grad()
    if not skipped_zero_advantage:
        loss.backward()
        clip_value = float(config.training.get("clip_value", 0.8))
        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

    rewards_grouped = rearrange(rewards.detach().cpu(), "(b g) -> b g", g=num_rollouts)
    reward_std_grouped = rewards_grouped.std(dim=1, unbiased=False)
    reward_std_min = float(rl_config.get("reward_std_min", 0.0))
    return {
        "loss": float(loss.detach().cpu()),
        "reward_mean": float(rewards.mean().detach().cpu()),
        "reward_max": float(rewards.max().detach().cpu()),
        "reward_min": float(rewards.min().detach().cpu()),
        "reward_group_std_mean": float(reward_std_grouped.mean().item()),
        "skipped_low_reward_std": float((reward_std_grouped <= reward_std_min).float().mean().item()),
        "pass_at_group": float((rewards_grouped.max(dim=1).values > 0).float().mean().item()),
        "advantage_abs_mean": float(advantages.abs().mean().detach().cpu()),
        "max_generate": max_generate,
        "zero_advantage": skipped_zero_advantage,
        "sample_hypothesis": hypotheses[0] if len(hypotheses) > 0 else "",
        "sample_reference": references[0] if len(references) > 0 else "",
    }


def init_wandb(config: OmegaConf, config_path: str):
    if not config.wandb.get("use", False):
        return None

    wandb_dir = config.wandb.get("dir", "./wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    run_id = config.wandb.get("id", "")
    kwargs = {
        "project": config.wandb.project_name,
        "name": config.wandb.get("name", None),
        "config": OmegaConf.to_container(config, resolve=True),
        "dir": wandb_dir,
    }
    if run_id:
        run = wandb.init(id=run_id, resume="must", allow_val_change=True, **kwargs)
    else:
        run = wandb.init(**kwargs)
    config.wandb.id = wandb.run.id
    if config.wandb.get("update_config_with_wandb_id", False):
        OmegaConf.save(config=config, f=config_path)
    print(f"\nLogging with WandB id: {wandb.run.id}\n")
    return run


def train(args: argparse.Namespace) -> None:
    args.config_path = args.config
    config = OmegaConf.load(args.config)
    os.makedirs(config.checkpointing.dir, exist_ok=True)

    tokenizer_kwargs = {}
    if "tokenizer_path" in config.training:
        tokenizer_kwargs["tokenizer_path"] = config.training.tokenizer_path
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)

    seed = int(config.training.get("random_seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config, model, optimizer, scheduler, seen_ids, step, epoch = load_or_initialize_model(
        config=config,
        tokenizer=tokenizer,
        device=device,
        reset_step=args.reset_step,
    )
    run = init_wandb(config, args.config_path)

    paired_data = lcasr.utils.audio_tools.load_json(config.data.path)
    normalizer = EnglishTextNormalizer() if EnglishTextNormalizer is not None else None
    dtype = get_dtype(config.training.get("dtype", "bfloat16"))
    max_steps = int(config.training.max_steps)
    chunk_size = int(config.audio_chunking.size)
    chunk_overlap = int(config.audio_chunking.get("overlap", 0))

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    pbar = tqdm(total=max_steps, initial=step, desc="RL updates")
    while step < max_steps:
        dataloader = VariableBatchSimpleDataloader(
            pairs=paired_data,
            tokenizer=tokenizer,
            batch_size=int(config.training.batch_size),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_workers=int(config.training.get("num_workers", args.num_workers)),
            pin_memory=bool(config.training.get("pin_memory", args.pin_memory)),
            prefetch=config.training.get("prefetch_factor", args.prefetch_factor),
            seen_ids=seen_ids,
            random_seed=seed + epoch,
        )

        for batch in dataloader:
            audio, audio_lengths, txt, ids = batch
            seen_ids.extend(ids)
            chunks = make_rl_chunks(
                audio=audio,
                audio_lengths=audio_lengths,
                txt=txt,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                pad_id=model.get_pad_id(),
                normalizer=normalizer,
            )
            if bool(config.rl.get("shuffle_chunks", True)):
                random.shuffle(chunks)

            for chunk in chunks:
                if step >= max_steps:
                    break
                metrics = rl_update(
                    model=model,
                    optimizer=optimizer,
                    chunk=chunk,
                    tokenizer=tokenizer,
                    config=config,
                    device=device,
                    dtype=dtype,
                    normalizer=normalizer,
                )
                step += 1
                scheduler_step(scheduler, step=step, max_steps=max_steps)
                lr = scheduler.get_last_lr()[0]
                metrics["learning_rate"] = lr
                metrics["step"] = step
                metrics["epoch"] = epoch
                metrics["algorithm"] = config.rl.algorithm
                pbar.update(1)
                pbar.set_postfix(
                    reward=f"{metrics['reward_mean']:.3f}",
                    pass_at_group=f"{metrics['pass_at_group']:.3f}",
                    loss=f"{metrics['loss']:.3f}",
                )

                if run is not None:
                    wandb.log(metrics, step=step)

                if step % int(config.checkpointing.save_every_n_steps) == 0:
                    save_model(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        podcast_step=step,
                        config=config,
                        seen_ids=seen_ids,
                        epoch=epoch,
                        other={"rl_metrics": metrics},
                    )

        epoch += 1
        seen_ids = reset_seen_ids(seen_ids=seen_ids, epoch=epoch - 1)

    save_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        podcast_step=step,
        config=config,
        seen_ids=seen_ids,
        epoch=epoch,
    )
    if run is not None:
        wandb.finish()


def validate_config(args: argparse.Namespace) -> None:
    config = OmegaConf.load(args.config)
    assert config.rl.algorithm in {"max_rl", "grpo"}
    assert int(config.rl.num_rollouts) > 1
    assert int(config.training.max_steps) > 0
    assert os.path.exists(config.data.path), f"data path missing: {config.data.path}"
    assert config.checkpointing.get("pretrained", None) is not None, "checkpointing.pretrained is required"
    checkpoint_path = resolve_checkpoint_path(config.checkpointing.pretrained)
    os.makedirs(config.checkpointing.dir, exist_ok=True)
    if config.wandb.get("use", False):
        os.makedirs(config.wandb.get("dir", "./wandb"), exist_ok=True)
    if args.validate_load_model:
        tokenizer_kwargs = {}
        if "tokenizer_path" in config.training:
            tokenizer_kwargs["tokenizer_path"] = config.training.tokenizer_path
        tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)
        checkpoint = load_checkpoint_config(checkpoint_path)
        config = merge_model_config(config, checkpoint)
        config.model.flash_attn = False
        model = load_model(config, tokenizer.vocab_size(), get_model_class(config=config))
        incompatible = load_model_state_compat(model, checkpoint["model"])
        print(
            "Checkpoint load validation passed "
            f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
        )
        if len(incompatible.missing_keys) > 0:
            print("Missing keys:", incompatible.missing_keys[:20])
        if len(incompatible.unexpected_keys) > 0:
            print("Unexpected keys:", incompatible.unexpected_keys[:20])
    print("Config validation passed")


def resolve_training_text(txt: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "word_timestamps" in txt:
        return txt["word_timestamps"]
    return txt["results"][-1]["alternatives"][0]["words"]


def smoke_rollout(args: argparse.Namespace) -> None:
    config = OmegaConf.load(args.config)
    config.wandb.use = False
    config.rl.num_rollouts = int(args.smoke_num_rollouts)
    config.rl.max_generate = int(args.smoke_max_generate)

    tokenizer_kwargs = {}
    if "tokenizer_path" in config.training:
        tokenizer_kwargs["tokenizer_path"] = config.training.tokenizer_path
    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer_kwargs)

    device = torch.device("cpu")
    config, model, optimizer, _, _, _, _ = load_or_initialize_model(
        config=config,
        tokenizer=tokenizer,
        device=device,
        reset_step=True,
    )
    model.train()

    paired_data = lcasr.utils.audio_tools.load_json(config.data.path)
    sample_id, sample = min(paired_data.items(), key=lambda item: item[1].get("duration", float("inf")))
    audio, txt = load_sample(sample)
    txt = resolve_training_text(txt)
    audio = audio.squeeze(0) if audio.ndim == 3 and audio.shape[0] == 1 else audio
    audio = audio.unsqueeze(0)
    audio_lengths = torch.LongTensor([audio.shape[-1]])
    normalizer = EnglishTextNormalizer() if EnglishTextNormalizer is not None else None

    chunks = make_rl_chunks(
        audio=audio,
        audio_lengths=audio_lengths,
        txt=[txt],
        tokenizer=tokenizer,
        chunk_size=int(config.audio_chunking.size),
        chunk_overlap=int(config.audio_chunking.get("overlap", 0)),
        pad_id=model.get_pad_id(),
        normalizer=normalizer,
    )
    if len(chunks) == 0:
        raise RuntimeError(f"no non-empty RL chunks produced for smoke sample {sample_id}")
    chunk = chunks[0]

    model_dtype = next(model.parameters()).dtype
    audio_chunk = chunk["audio"].to(device, dtype=model_dtype)
    audio_chunk_lengths = chunk["audio_lengths"].to(device)
    num_rollouts = int(config.rl.num_rollouts)
    bos_id = model.get_bos_id()
    eos_id = model.get_eos_id()
    pad_id = model.get_pad_id()

    actions, _ = sample_rollouts(
        model=model,
        audio=audio_chunk,
        audio_lengths=audio_chunk_lengths,
        max_generate=int(config.rl.max_generate),
        num_rollouts=num_rollouts,
        temperature=float(config.rl.temperature),
        bos_id=bos_id,
        eos_id=eos_id,
    )
    hypotheses = [
        decode_actions(action, tokenizer=tokenizer, eos_id=eos_id, pad_id=pad_id, normalizer=normalizer)
        for action in actions
    ]
    references = [reference for reference in chunk["references"] for _ in range(num_rollouts)]
    rewards = compute_rewards(
        hypotheses=hypotheses,
        references=references,
        algorithm=config.rl.algorithm,
        reward_config=config.rl,
    )
    advantages = compute_advantages(
        rewards=rewards,
        group_size=num_rollouts,
        algorithm=config.rl.algorithm,
        eps=float(config.rl.get("advantage_eps", 1e-6)),
        min_group_std=float(config.rl.get("reward_std_min", 0.0)),
    )

    optimizer.zero_grad()
    was_training = model.training
    model.eval()
    try:
        logprobs = sequence_logprobs(
            model=model,
            audio=audio_chunk,
            audio_lengths=audio_chunk_lengths,
            actions=actions,
            num_rollouts=num_rollouts,
            bos_id=bos_id,
            pad_id=pad_id,
        )
    finally:
        if was_training:
            model.train()
    loss = -logprobs.mean()
    loss.backward()
    optimizer.step()

    print(
        "Smoke rollout passed "
        f"(sample={sample_id}, chunk_batch={audio_chunk.shape[0]}, rollouts={len(actions)}, "
        f"max_generate={config.rl.max_generate}, loss={float(loss.detach().cpu()):.4f}, "
        f"reward_mean={float(rewards.mean()):.4f}, advantage_abs_mean={float(advantages.abs().mean()):.4f})"
    )


def self_test() -> None:
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    max_rl_adv = compute_advantages(rewards, group_size=3, algorithm="max_rl", eps=1e-6)
    grpo_adv = compute_advantages(rewards, group_size=3, algorithm="grpo", eps=1e-6)
    assert torch.isfinite(max_rl_adv).all()
    assert torch.isfinite(grpo_adv).all()
    assert max_rl_adv[:3].sum().abs() < 1e-5
    assert grpo_adv[:3].sum().abs() < 1e-5
    rewards = perfect_wer_rewards(["hello world", "hello"], ["hello world", "hello world"])
    assert rewards.tolist() == [1.0, 0.0]
    rewards = weighted_error_rewards(
        ["hello world", "hello world now", "hello"],
        ["hello world", "hello world", "hello world"],
        wer_weight=0.7,
        cer_weight=0.3,
        reward_offset=1.0,
        reward_scale=1.0,
        reward_min=0.0,
        reward_max=None,
        reward_positive_threshold=None,
    )
    assert rewards[0].item() == 1.0
    assert rewards[1].item() < 1.0
    assert rewards[2].item() < 1.0

    low_var_adv = compute_advantages(
        torch.tensor([0.50, 0.51, 0.50, 0.90, 0.10, 0.50]),
        group_size=3,
        algorithm="grpo",
        eps=1e-6,
        min_group_std=0.02,
    )
    assert low_var_adv[:3].abs().sum() == 0
    assert low_var_adv[3:].abs().sum() > 0

    class ToyTokenizer:
        def encode(self, text):
            return [1 for token in text.split() if token]

        def decode(self, tokens):
            return " ".join("tok" for _ in tokens)

    chunks = make_rl_chunks(
        audio=torch.zeros(2, 80, 8),
        audio_lengths=torch.LongTensor([4, 8]),
        txt=[
            [{"start": 0.0, "end": 0.01, "text": "short"}],
            [
                {"start": 0.0, "end": 0.01, "text": "longone"},
                {"start": 0.05, "end": 0.07, "text": "longtwo"},
            ],
        ],
        tokenizer=ToyTokenizer(),
        chunk_size=4,
        chunk_overlap=0,
        pad_id=0,
        normalizer=None,
    )
    assert len(chunks) == 2
    assert chunks[0]["audio_lengths"].tolist() == [4, 4]
    assert chunks[1]["audio_lengths"].tolist() == [4]
    assert all(chunk["audio_lengths"].min().item() > 0 for chunk in chunks)

    class ToyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(4, 3)
            self.proj = torch.nn.Linear(3, 4)

        def forward(self, tokens, a_hidden, a_lengths, text_lengths=None, cache=None):
            logits = self.proj(self.embed(tokens))
            logits[..., 2] = logits[..., 2] + 5.0
            return {"logits": logits, "kv_cache": None}

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model_decoder = ToyDecoder()

        def forward(self, audio_signal, length=None):
            batch = audio_signal.shape[0]
            return {
                "a_hidden": torch.ones(batch, 2, 3, device=audio_signal.device),
                "length": torch.full((batch,), 2, dtype=torch.long, device=audio_signal.device),
            }

        def get_bos_id(self):
            return 0

        def get_eos_id(self):
            return 0

        def get_pad_id(self):
            return 0

    toy = ToyModel()
    audio = torch.zeros(2, 80, 8)
    audio_lengths = torch.full((2,), 8, dtype=torch.long)
    actions, _ = sample_rollouts(
        model=toy,
        audio=audio,
        audio_lengths=audio_lengths,
        max_generate=3,
        num_rollouts=2,
        temperature=1.0,
        bos_id=0,
        eos_id=0,
    )
    assert len(actions) == 4
    logprobs = sequence_logprobs(
        model=toy,
        audio=audio,
        audio_lengths=audio_lengths,
        actions=actions,
        num_rollouts=2,
        bos_id=0,
        pad_id=0,
    )
    assert logprobs.shape == (4,)
    (-logprobs.mean()).backward()
    assert toy.language_model_decoder.proj.weight.grad is not None
    print("Self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", type=str, required=False, help="path to config file")
    parser.add_argument("-reset_step", "--reset_step", action="store_true", help="start from pretrained even if output checkpoints exist")
    parser.add_argument("-num_workers", "--num_workers", type=int, default=0)
    parser.add_argument("-pin_memory", "--pin_memory", action="store_true")
    parser.add_argument("-prefetch", "--prefetch_factor", type=int, default=None)
    parser.add_argument("--validate_config_only", action="store_true")
    parser.add_argument("--validate_load_model", action="store_true")
    parser.add_argument("--smoke_rollout", action="store_true")
    parser.add_argument("--smoke_max_generate", type=int, default=4)
    parser.add_argument("--smoke_num_rollouts", type=int, default=2)
    parser.add_argument("--self_test", action="store_true")
    parsed = parser.parse_args()

    if parsed.self_test:
        self_test()
    else:
        if parsed.config is None:
            raise ValueError("--config is required unless --self_test is set")
        if parsed.validate_config_only:
            validate_config(parsed)
        elif parsed.smoke_rollout:
            smoke_rollout(parsed)
        else:
            train(parsed)
