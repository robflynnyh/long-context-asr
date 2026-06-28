#!/usr/bin/env python3
"""ROB-319 low-rank weight deletion search for long-context ASR.

This is an issue-scoped experiment runner. It perturbs only selected
SCConformerXL feed-forward and convolution weights in memory, evaluates
Earnings recordings in deterministic blocks, and records enough artifacts to
reproduce the chosen low-rank deletion direction without modifying source
checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from whisper.normalizers import EnglishTextNormalizer

import lcasr
from lcasr.decoding.greedy import GreedyCTCDecoder
from lcasr.eval.buffered_transcription import fetch_logits as buffered_eval
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.wer import word_error_rate_detail
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import get_model_class, load_model
from lcasr.utils.omegaconf import OmegaConf


NORMALIZE = EnglishTextNormalizer()
DEFAULT_CONFIG = Path("symphony/configs/rob319_eggroll_deletion_search.yaml")
WANDB_REQUIRED_KEYS = {
    "context_score",
    "clean_gain",
    "damaged_gain",
    "retained_gain",
    "mean_damage",
    "collapse_penalty",
    "ordering_penalty",
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    seq_len: int
    repeat: int
    path: str
    overlap_ratio: float = 0.875
    model_class: str = "SCConformerXL"


@dataclass(frozen=True)
class EarningsRecord:
    id: str
    audio: str
    text: str
    transcript_key: str


@dataclass(frozen=True)
class Block:
    block_id: int
    kind: str
    records: Tuple[EarningsRecord, ...]

    @property
    def recording_ids(self) -> List[str]:
        return [record.id for record in self.records]


@dataclass(frozen=True)
class TargetTensor:
    name: str
    shape: Tuple[int, ...]
    numel: int
    group: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    pair_id: int
    sign: int


@dataclass(frozen=True)
class BlockWers:
    block_id: int
    recording_ids: Tuple[str, ...]
    wers: Dict[str, float]


@dataclass
class ModelBundle:
    spec: ModelSpec
    model: torch.nn.Module
    config: Any
    eval_args: SimpleNamespace
    decoder: Optional[GreedyCTCDecoder]
    tokenizer: Any
    device: torch.device


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


class CsvWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        serialised = {key: row.get(key) for key in self.fieldnames}
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(serialised)


class WandbLogger:
    def __init__(self, config: Mapping[str, Any], output_root: Path, disabled: bool) -> None:
        self.run = None
        self.disabled = disabled
        wandb_cfg = config.get("wandb", {})
        if disabled or not wandb_cfg.get("enabled", True):
            return

        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime env
            print(f"W&B unavailable; continuing without online logging: {exc}", file=sys.stderr)
            self.disabled = True
            return

        wandb_dir = Path(str(wandb_cfg.get("dir", output_root / "wandb")))
        wandb_dir.mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=wandb_cfg.get("project", "long-context-asr"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name", f"rob319-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"),
            group=wandb_cfg.get("group", "ROB-319"),
            tags=list(wandb_cfg.get("tags", ["ROB-319", "eggroll-deletion"])),
            mode=wandb_cfg.get("mode", "online"),
            dir=str(wandb_dir),
            config=to_jsonable(config),
        )

    def log(self, row: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        self.run.log(to_jsonable(row))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return to_jsonable(vars(value))
    return value


def stable_seed(base_seed: int, tensor_name: str, pair_id: int) -> int:
    payload = f"{base_seed}:{tensor_name}:{pair_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def iter_candidates(num_pairs: int) -> Iterator[Candidate]:
    for pair_id in range(num_pairs):
        yield Candidate(candidate_id=pair_id, pair_id=pair_id, sign=1)
    for pair_id in range(num_pairs):
        yield Candidate(candidate_id=num_pairs + pair_id, pair_id=pair_id, sign=-1)


def low_rank_delta(
    shape: Sequence[int],
    *,
    tensor_name: str,
    pair_id: int,
    rank: int,
    base_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if len(shape) < 2:
        raise ValueError(f"Low-rank perturbations require at least 2D tensors, got {shape}")
    if rank < 1:
        raise ValueError("rank must be positive")

    out_dim = int(shape[0])
    flat_in_dim = int(math.prod(shape[1:]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(base_seed, tensor_name, pair_id))
    a = torch.randn(out_dim, rank, generator=generator, dtype=torch.float32)
    b = torch.randn(flat_in_dim, rank, generator=generator, dtype=torch.float32)
    delta = (a @ b.T) / math.sqrt(rank)
    return delta.reshape(tuple(shape)).to(device=device, dtype=dtype)


def target_group(name: str) -> str:
    if ".ff1." in name:
        return "ff1"
    if ".ff2." in name:
        return "ff2"
    if ".conv." in name:
        return "conv"
    return "unknown"


def should_target_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    lname = name.lower()
    if not name.startswith("layers."):
        return False
    if not name.endswith(".weight"):
        return False
    if parameter.ndim < 2:
        return False

    excluded_fragments = (
        "attend",
        "decoder",
        "norm",
        "bias",
        "subsampling",
        "rotary",
        "pos",
        "embedding",
        "emb",
    )
    if any(fragment in lname for fragment in excluded_fragments):
        return False

    is_ff = (".ff1." in name or ".ff2." in name) and (
        name.endswith(".fc1.weight") or name.endswith(".fc2.weight")
    )
    is_conv = ".conv." in name and (
        name.endswith(".pointwise_conv1.weight")
        or name.endswith(".depthwise_conv.weight")
        or name.endswith(".pointwise_conv2.weight")
    )
    return bool(is_ff or is_conv)


def select_target_tensors(model: torch.nn.Module) -> List[TargetTensor]:
    targets: List[TargetTensor] = []
    for name, parameter in model.named_parameters():
        if should_target_parameter(name, parameter):
            targets.append(
                TargetTensor(
                    name=name,
                    shape=tuple(int(dim) for dim in parameter.shape),
                    numel=int(parameter.numel()),
                    group=target_group(name),
                )
            )
    return targets


def get_parameter_map(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    return dict(model.named_parameters())


def validate_shared_targets(targets_by_model: Mapping[str, Sequence[TargetTensor]]) -> List[TargetTensor]:
    if not targets_by_model:
        raise ValueError("No models loaded")
    labels = list(targets_by_model)
    reference = list(targets_by_model[labels[0]])
    if not reference:
        raise ValueError("Target selector matched no tensors")

    reference_pairs = [(target.name, target.shape) for target in reference]
    for label in labels[1:]:
        cur_pairs = [(target.name, target.shape) for target in targets_by_model[label]]
        if cur_pairs != reference_pairs:
            ref_set = set(reference_pairs)
            cur_set = set(cur_pairs)
            missing = sorted(ref_set - cur_set)[:20]
            extra = sorted(cur_set - ref_set)[:20]
            raise ValueError(
                f"Target tensor mismatch for {label}: missing={missing}, extra={extra}"
            )
    return reference


@contextlib.contextmanager
def applied_candidate_perturbation(
    model: torch.nn.Module,
    targets: Sequence[TargetTensor],
    *,
    candidate: Candidate,
    rank: int,
    sigma: float,
    base_seed: int,
) -> Iterator[None]:
    if sigma == 0:
        yield
        return

    parameter_map = get_parameter_map(model)
    applied: List[Tuple[TargetTensor, torch.device, torch.dtype]] = []
    with torch.no_grad():
        for target in targets:
            parameter = parameter_map[target.name]
            delta = low_rank_delta(
                target.shape,
                tensor_name=target.name,
                pair_id=candidate.pair_id,
                rank=rank,
                base_seed=base_seed,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(delta, alpha=float(sigma * candidate.sign))
            applied.append((target, parameter.device, parameter.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for target, device, dtype in reversed(applied):
                parameter = parameter_map[target.name]
                delta = low_rank_delta(
                    target.shape,
                    tensor_name=target.name,
                    pair_id=candidate.pair_id,
                    rank=rank,
                    base_seed=base_seed,
                    device=device,
                    dtype=dtype,
                )
                parameter.sub_(delta, alpha=float(sigma * candidate.sign))


def combined_delta(
    target: TargetTensor,
    pair_weights: Mapping[int, float],
    *,
    rank: int,
    base_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.zeros(target.shape, device=device, dtype=torch.float32)
    pair_count = max(len(pair_weights), 1)
    for pair_id, pair_weight in sorted(pair_weights.items()):
        if pair_weight == 0:
            continue
        delta = low_rank_delta(
            target.shape,
            tensor_name=target.name,
            pair_id=pair_id,
            rank=rank,
            base_seed=base_seed,
            device=device,
            dtype=torch.float32,
        )
        output.add_(delta, alpha=float(pair_weight) / pair_count)
    return output.to(dtype=dtype)


@contextlib.contextmanager
def applied_combined_perturbation(
    model: torch.nn.Module,
    targets: Sequence[TargetTensor],
    pair_weights: Mapping[int, float],
    *,
    rank: int,
    eta: float,
    base_seed: int,
) -> Iterator[None]:
    if eta == 0 or not pair_weights:
        yield
        return

    parameter_map = get_parameter_map(model)
    applied: List[Tuple[TargetTensor, torch.device, torch.dtype]] = []
    with torch.no_grad():
        for target in targets:
            parameter = parameter_map[target.name]
            delta = combined_delta(
                target,
                pair_weights,
                rank=rank,
                base_seed=base_seed,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(delta, alpha=float(eta))
            applied.append((target, parameter.device, parameter.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for target, device, dtype in reversed(applied):
                parameter = parameter_map[target.name]
                delta = combined_delta(
                    target,
                    pair_weights,
                    rank=rank,
                    base_seed=base_seed,
                    device=device,
                    dtype=dtype,
                )
                parameter.sub_(delta, alpha=float(eta))


def preprocess_transcript(text: str) -> str:
    text = text.lower()
    for token in (
        "<silence>",
        "<inaudible>",
        "<laugh>",
        "<noise>",
        "<affirmative>",
        "<crosstalk>",
    ):
        text = text.replace(token, "")
    for old, new in (("...", ""), ("…", ""), (",", ""), ("-", " "), (".", ""), ("?", "")):
        text = text.replace(old, new)
    text = " ".join(text.split())
    return NORMALIZE(text).lower()


def resolve_transcript_key(stem: str, transcripts: Mapping[str, str]) -> str:
    if stem in transcripts:
        return stem
    candidates = [
        key
        for key in transcripts
        if stem.startswith(key) or key.startswith(stem) or key.lower() in stem.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"No unique transcript key for audio stem {stem!r}; candidates={candidates[:10]}")


def load_earnings_records(config: Mapping[str, Any]) -> Tuple[List[EarningsRecord], Dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    root = Path(str(dataset_cfg.get("root", "/store/store4/data/earnings-22")))
    split = str(dataset_cfg.get("split", "train"))
    transcripts_path = Path(str(dataset_cfg.get("transcripts", root / "full_transcripts.json")))
    if not transcripts_path.exists():
        raise FileNotFoundError(f"Earnings transcript file not found: {transcripts_path}")
    transcripts = json.loads(transcripts_path.read_text(encoding="utf-8"))
    if not isinstance(transcripts, dict):
        raise TypeError(f"Expected dict transcripts in {transcripts_path}")

    split_to_dir = {
        "train": "train",
        "test": "test_original",
        "dev": "dev_original",
    }
    if split not in split_to_dir:
        raise ValueError(f"Unsupported Earnings split {split!r}; expected one of {sorted(split_to_dir)}")
    audio_dir = Path(str(dataset_cfg.get("audio_dir", root / split_to_dir[split])))
    if not audio_dir.exists():
        raise FileNotFoundError(f"Earnings audio directory not found: {audio_dir}")

    records: List[EarningsRecord] = []
    transcript_aliases: Dict[str, str] = {}
    for audio_path in sorted(audio_dir.glob("*.mp3")):
        transcript_key = resolve_transcript_key(audio_path.stem, transcripts)
        transcript_aliases[audio_path.stem] = transcript_key
        records.append(
            EarningsRecord(
                id=audio_path.stem,
                audio=str(audio_path),
                text=str(transcripts[transcript_key]),
                transcript_key=transcript_key,
            )
        )

    expected_count = dataset_cfg.get("expected_recordings")
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(f"Expected {expected_count} Earnings records, found {len(records)}")
    if not records:
        raise ValueError(f"No .mp3 files found in {audio_dir}")

    metadata = {
        "root": str(root),
        "split": split,
        "audio_dir": str(audio_dir),
        "transcripts": str(transcripts_path),
        "record_count": len(records),
        "transcript_aliases": transcript_aliases,
    }
    return records, metadata


def make_blocks(records: Sequence[EarningsRecord], block_size: int, search_fraction: float) -> Tuple[List[Block], List[Block]]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    blocks = [
        records[index : index + block_size]
        for index in range(0, len(records), block_size)
        if len(records[index : index + block_size]) == block_size
    ]
    if len(blocks) < 2:
        raise ValueError("Need at least two full blocks for search/validation split")
    search_count = int(math.floor(len(blocks) * search_fraction))
    search_count = min(max(search_count, 1), len(blocks) - 1)
    search_blocks = [Block(block_id=i, kind="search", records=tuple(block)) for i, block in enumerate(blocks[:search_count])]
    validation_blocks = [
        Block(block_id=i + search_count, kind="validation", records=tuple(block))
        for i, block in enumerate(blocks[search_count:])
    ]
    return search_blocks, validation_blocks


def load_tokenizer(tokenizer_path: Optional[str]) -> Any:
    kwargs = {"tokenizer_path": tokenizer_path} if tokenizer_path else {}
    return lcasr.utils.audio_tools.load_tokenizer(**kwargs)


def device_for_index(index: int, requested: Sequence[str]) -> torch.device:
    if requested:
        return torch.device(requested[index % len(requested)])
    if torch.cuda.is_available():
        count = max(torch.cuda.device_count(), 1)
        return torch.device(f"cuda:{index % count}")
    return torch.device("cpu")


def prepare_eval_args(
    config: Any,
    model_spec: ModelSpec,
    run_cfg: Mapping[str, Any],
) -> SimpleNamespace:
    eval_mode = str(run_cfg.get("evaluation_mode", "windowed_attention"))
    seq_len = int(model_spec.seq_len)
    overlap = int(seq_len * float(model_spec.overlap_ratio))
    max_sequence_length = int(run_cfg.get("max_sequence_length", 3_600_000))

    if run_cfg.get("disable_flash_attention", False):
        config.model.flash_attn = False

    if eval_mode == "windowed_attention":
        subsample_factor = int(config.model.get("subsampling_factor", 8))
        window_size = run_cfg.get("window_size")
        if window_size is None:
            window_size = seq_len // subsample_factor // 2
        config.model.attention_window_size = int(window_size)
        eval_seq_len = max_sequence_length
    else:
        eval_seq_len = seq_len

    return SimpleNamespace(
        config=config,
        checkpoint=model_spec.path,
        split=run_cfg.get("split", "train"),
        seq_len=eval_seq_len,
        training_seq_len=seq_len,
        overlap=overlap,
        dataset="earnings22",
        model_class=model_spec.model_class,
        evaluation_mode=eval_mode,
        max_sequence_length=max_sequence_length,
        verbose=bool(run_cfg.get("verbose", False)),
        transcribe_kwargs=dict(run_cfg.get("transcribe_kwargs", {}) or {}),
    )


def load_model_bundle(
    spec: ModelSpec,
    *,
    device: torch.device,
    tokenizer: Any,
    run_cfg: Mapping[str, Any],
) -> ModelBundle:
    checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    eval_args = prepare_eval_args(model_config, spec, run_cfg)
    model_class = get_model_class({"model_class": model_config.get("model_class", spec.model_class)})
    model = load_model(model_config, tokenizer.vocab_size(), model_class=model_class)
    model.print_total_params()
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    print(
        f"Loaded {spec.label} from {spec.path} on {device}; "
        f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
    )
    model.device = device
    model = model.to(device)
    model.eval()
    decoder = None
    if not hasattr(model, "transcribe"):
        decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=model.decoder.num_classes - 1)
    return ModelBundle(
        spec=spec,
        model=model,
        config=model_config,
        eval_args=eval_args,
        decoder=decoder,
        tokenizer=tokenizer,
        device=device,
    )


def load_model_specs(config: Mapping[str, Any], checkpoint_root_override: Optional[str]) -> List[ModelSpec]:
    models = []
    checkpoint_root = Path(str(checkpoint_root_override or config.get("paths", {}).get("checkpoint_root", "")))
    for item in config.get("models", []):
        path = str(item.get("path", ""))
        if not path:
            path = str(checkpoint_root / f"n_seq_sched_{int(item['seq_len'])}_rp_{int(item['repeat'])}" / "step_105360.pt")
        models.append(
            ModelSpec(
                label=str(item["label"]),
                seq_len=int(item["seq_len"]),
                repeat=int(item.get("repeat", 1)),
                path=path,
                overlap_ratio=float(item.get("overlap_ratio", config.get("search", {}).get("overlap_ratio", 0.875))),
                model_class=str(item.get("model_class", "SCConformerXL")),
            )
        )
    if [model.label for model in models] != ["short", "medium", "long"]:
        labels = [model.label for model in models]
        raise ValueError(f"Expected model labels [short, medium, long], got {labels}")
    return models


def process_record(record: EarningsRecord) -> Tuple[torch.Tensor, str]:
    return processing_chain(record.audio), preprocess_transcript(record.text)


def decode_record(bundle: ModelBundle, record: EarningsRecord, use_tqdm: bool) -> Tuple[str, str]:
    audio_spec, gold_text = process_record(record)
    if hasattr(bundle.model, "transcribe"):
        kwargs = dict(getattr(bundle.eval_args, "transcribe_kwargs", {}) or {})
        kwargs.setdefault("verbose", False)
        with torch.no_grad():
            all_text = bundle.model.transcribe(
                audio_spec,
                bundle.tokenizer,
                device=bundle.device,
                max_sequence_length=bundle.eval_args.seq_len,
                **kwargs,
            )
        out = NORMALIZE(all_text).lower().strip()
    else:
        eval_fn = moving_average_eval
        if bundle.eval_args.evaluation_mode == "buffered":
            eval_fn = buffered_eval
        logits = eval_fn(
            args=bundle.eval_args,
            model=bundle.model,
            spec=audio_spec,
            seq_len=bundle.eval_args.seq_len,
            overlap=bundle.eval_args.overlap,
            tokenizer=bundle.tokenizer,
            use_tqdm=use_tqdm,
        )
        assert bundle.decoder is not None
        out_text = bundle.decoder(torch.as_tensor(logits))
        out = NORMALIZE(out_text).lower().strip()
    return out, gold_text


def evaluate_model_on_block(bundle: ModelBundle, block: Block, use_tqdm: bool) -> Dict[str, Any]:
    all_texts: List[str] = []
    all_golds: List[str] = []
    per_recording: List[Dict[str, Any]] = []
    for record in block.records:
        hyp, ref = decode_record(bundle, record, use_tqdm=use_tqdm)
        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
            hypotheses=[hyp],
            references=[ref],
        )
        per_recording.append(
            {
                "recording": record.id,
                "wer": wer,
                "words": words,
                "ins_rate": ins_rate,
                "del_rate": del_rate,
                "sub_rate": sub_rate,
            }
        )
        all_texts.append(hyp)
        all_golds.append(ref)

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
        hypotheses=all_texts,
        references=all_golds,
    )
    return {
        "wer": float(wer),
        "words": int(words),
        "ins_rate": float(ins_rate),
        "del_rate": float(del_rate),
        "sub_rate": float(sub_rate),
        "per_recording": per_recording,
    }


def evaluate_block_for_models(
    bundles: Mapping[str, ModelBundle],
    block: Block,
    *,
    use_tqdm: bool,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    wers: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}
    for label, bundle in bundles.items():
        result = evaluate_model_on_block(bundle, block, use_tqdm=use_tqdm)
        wers[label] = float(result["wer"])
        diagnostics[label] = result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return wers, diagnostics


def score_candidate(
    clean: Mapping[str, float],
    damaged: Mapping[str, float],
    score_cfg: Mapping[str, Any],
) -> Dict[str, float]:
    eps = float(score_cfg.get("eps", 1e-6))
    beta = float(score_cfg.get("beta", 0.25))
    gamma = float(score_cfg.get("gamma", 1.0))
    lamb = float(score_cfg.get("lambda", score_cfg.get("ordering_lambda", 1.0)))
    useful_damage_cap = float(score_cfg.get("useful_damage_cap", 0.15))
    collapse_wer = float(score_cfg.get("collapse_wer", 0.95))
    collapse_damage = float(score_cfg.get("collapse_damage", 0.45))

    clean_gain = float(clean["short"] - clean["long"])
    damaged_gain = float(damaged["short"] - damaged["long"])
    retained_gain = damaged_gain / max(clean_gain, eps)
    damage_values = [float(damaged[label] - clean[label]) for label in ("short", "medium", "long")]
    mean_damage = sum(damage_values) / len(damage_values)
    useful_damage = max(0.0, min(mean_damage, useful_damage_cap)) / max(useful_damage_cap, eps)

    max_damaged_wer = max(float(damaged[label]) for label in ("short", "medium", "long"))
    collapse_penalty = 0.0
    if max_damaged_wer > collapse_wer:
        collapse_penalty += (max_damaged_wer - collapse_wer) / max(1.0 - collapse_wer, eps)
    if mean_damage > collapse_damage:
        collapse_penalty += (mean_damage - collapse_damage) / max(collapse_damage, eps)

    ordering_penalty = 0.0
    if damaged["long"] >= damaged["short"]:
        ordering_penalty += 1.0
    if bool(score_cfg.get("penalize_medium_ordering", True)):
        if damaged["long"] > damaged["medium"]:
            ordering_penalty += 0.5
        if damaged["medium"] > damaged["short"]:
            ordering_penalty += 0.5

    context_score = retained_gain + beta * useful_damage - gamma * collapse_penalty - lamb * ordering_penalty
    return {
        "context_score": float(context_score),
        "clean_gain": float(clean_gain),
        "damaged_gain": float(damaged_gain),
        "retained_gain": float(retained_gain),
        "mean_damage": float(mean_damage),
        "useful_damage": float(useful_damage),
        "collapse_penalty": float(collapse_penalty),
        "ordering_penalty": float(ordering_penalty),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_block_manifest(path: Path, search_blocks: Sequence[Block], validation_blocks: Sequence[Block]) -> None:
    payload = []
    for block in list(search_blocks) + list(validation_blocks):
        payload.append(
            {
                "block_id": block.block_id,
                "kind": block.kind,
                "recordings": [
                    {
                        "id": record.id,
                        "audio": record.audio,
                        "transcript_key": record.transcript_key,
                    }
                    for record in block.records
                ],
            }
        )
    write_json(path, payload)


def write_artifact_index(
    path: Path,
    *,
    run_dir: Path,
    config_path: Path,
    kept_artifacts: Sequence[Path],
    notes: Sequence[str],
) -> None:
    lines = [
        "# ROB-319 Artifact Index",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Source config: `{config_path}`",
        "- Source checkpoints are read-only and are never overwritten.",
        "",
        "## Kept Artifacts",
        "",
    ]
    for artifact in kept_artifacts:
        lines.append(f"- `{artifact}`")
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_combined_checkpoint(
    bundle: ModelBundle,
    source_checkpoint_path: str,
    output_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model"] = {key: value.detach().cpu() for key, value in bundle.model.state_dict().items()}
    checkpoint.setdefault("metadata", {})
    checkpoint["metadata"]["ROB-319"] = to_jsonable(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def read_config(path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    paths = config.setdefault("paths", {})
    dataset = config.setdefault("dataset", {})
    search = config.setdefault("search", {})
    wandb_cfg = config.setdefault("wandb", {})

    if args.artifact_root is not None:
        paths["artifact_root"] = args.artifact_root
    if args.checkpoint_root is not None:
        paths["checkpoint_root"] = args.checkpoint_root
    if args.earnings_root is not None:
        dataset["root"] = args.earnings_root
    if args.split is not None:
        dataset["split"] = args.split
    if args.num_pairs is not None:
        search["num_pairs"] = args.num_pairs
    if args.rank is not None:
        search["rank"] = args.rank
    if args.sigma is not None:
        search["sigma"] = args.sigma
    if args.eta is not None:
        search["eta"] = args.eta
    if args.max_search_blocks is not None:
        search["max_search_blocks"] = args.max_search_blocks
    if args.max_validation_blocks is not None:
        search["max_validation_blocks"] = args.max_validation_blocks
    if args.wandb_mode is not None:
        wandb_cfg["mode"] = args.wandb_mode
    if args.disable_wandb:
        wandb_cfg["enabled"] = False
    if args.save_combined_checkpoints:
        search["save_combined_checkpoints"] = True
    return config


def load_bundles_and_targets(
    config: Mapping[str, Any],
    model_specs: Sequence[ModelSpec],
    *,
    devices: Sequence[str],
    list_targets_only: bool,
) -> Tuple[Dict[str, ModelBundle], List[TargetTensor]]:
    tokenizer_path = config.get("paths", {}).get("tokenizer_path")
    tokenizer = load_tokenizer(str(tokenizer_path) if tokenizer_path else None)
    run_cfg = config.get("evaluation", {})
    bundles: Dict[str, ModelBundle] = {}
    targets_by_model: Dict[str, List[TargetTensor]] = {}
    for index, spec in enumerate(model_specs):
        if not Path(spec.path).exists():
            raise FileNotFoundError(f"Checkpoint for {spec.label} does not exist: {spec.path}")
        bundle = load_model_bundle(
            spec,
            device=device_for_index(index, devices),
            tokenizer=tokenizer,
            run_cfg=run_cfg,
        )
        targets = select_target_tensors(bundle.model)
        targets_by_model[spec.label] = targets
        bundles[spec.label] = bundle
        print(f"{spec.label}: matched {len(targets)} target tensors")
        if list_targets_only:
            for target in targets:
                print(f"  {target.name} shape={target.shape} group={target.group}")

    shared_targets = validate_shared_targets(targets_by_model)
    return bundles, shared_targets


def run_determinism_check(
    targets: Sequence[TargetTensor],
    *,
    rank: int,
    base_seed: int,
) -> Dict[str, Any]:
    if not targets:
        raise ValueError("No targets available for determinism check")
    target = targets[0]
    delta_a = low_rank_delta(
        target.shape,
        tensor_name=target.name,
        pair_id=0,
        rank=rank,
        base_seed=base_seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    delta_b = low_rank_delta(
        target.shape,
        tensor_name=target.name,
        pair_id=0,
        rank=rank,
        base_seed=base_seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    delta_other = low_rank_delta(
        target.shape,
        tensor_name=target.name,
        pair_id=1,
        rank=rank,
        base_seed=base_seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return {
        "target": target.name,
        "same_pair_equal": bool(torch.equal(delta_a, delta_b)),
        "different_pair_equal": bool(torch.equal(delta_a, delta_other)),
        "positive_negative_antithetic": True,
        "delta_mean": float(delta_a.mean().item()),
        "delta_std": float(delta_a.std(unbiased=False).item()),
    }


def run_search(config: Mapping[str, Any], config_path: Path, args: argparse.Namespace) -> int:
    run_id = args.run_id or time.strftime("rob319-eggroll-deletion-%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = Path(str(config.get("paths", {}).get("artifact_root")))
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_specs = load_model_specs(config, args.checkpoint_root)
    records, dataset_metadata = load_earnings_records(config)
    dataset_cfg = config.get("dataset", {})
    search_cfg = config.get("search", {})
    score_cfg = config.get("score", {})

    search_blocks, validation_blocks = make_blocks(
        records,
        block_size=int(dataset_cfg.get("block_size", 5)),
        search_fraction=float(dataset_cfg.get("search_fraction", 0.8)),
    )
    max_search_blocks = search_cfg.get("max_search_blocks")
    max_validation_blocks = search_cfg.get("max_validation_blocks")
    if max_search_blocks is not None:
        search_blocks = search_blocks[: int(max_search_blocks)]
    if max_validation_blocks is not None:
        validation_blocks = validation_blocks[: int(max_validation_blocks)]

    if not search_blocks:
        raise ValueError("No search blocks selected")
    if not validation_blocks and not args.skip_validation:
        raise ValueError("No validation blocks selected")

    write_json(run_dir / "config_resolved.json", config)
    write_json(run_dir / "model_specs.json", [asdict(model) for model in model_specs])
    write_json(run_dir / "dataset_metadata.json", dataset_metadata)
    save_block_manifest(run_dir / "blocks.json", search_blocks, validation_blocks)

    devices = [item.strip() for item in str(args.devices or "").split(",") if item.strip()]
    bundles, targets = load_bundles_and_targets(
        config,
        model_specs,
        devices=devices,
        list_targets_only=args.list_targets,
    )
    write_json(run_dir / "target_tensors.json", [asdict(target) for target in targets])
    target_summary = {
        "count": len(targets),
        "numel": sum(target.numel for target in targets),
        "by_group": {
            group: sum(1 for target in targets if target.group == group)
            for group in sorted({target.group for target in targets})
        },
    }
    write_json(run_dir / "target_summary.json", target_summary)

    rank = int(search_cfg.get("rank", 4))
    sigma = float(search_cfg.get("sigma", 1e-4))
    eta = float(search_cfg.get("eta", 1e-4))
    base_seed = int(search_cfg.get("base_seed", 319))
    num_pairs = int(search_cfg.get("num_pairs", 32))
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")

    determinism = run_determinism_check(targets, rank=rank, base_seed=base_seed)
    write_json(run_dir / "determinism_check.json", determinism)
    if args.determinism_check or args.list_targets:
        if args.list_targets and not args.run_after_listing:
            return 0
        if args.determinism_check and not args.run_after_listing:
            return 0

    candidate_writer = JsonlWriter(run_dir / "candidate_metrics.jsonl")
    clean_writer = JsonlWriter(run_dir / "clean_metrics.jsonl")
    validation_writer = JsonlWriter(run_dir / "validation_metrics.jsonl")
    candidate_csv = CsvWriter(
        run_dir / "candidate_metrics.csv",
        [
            "block_id",
            "candidate_id",
            "pair_id",
            "sign",
            "context_score",
            "clean_gain",
            "damaged_gain",
            "retained_gain",
            "mean_damage",
            "useful_damage",
            "collapse_penalty",
            "ordering_penalty",
            "short_clean_wer",
            "medium_clean_wer",
            "long_clean_wer",
            "short_candidate_wer",
            "medium_candidate_wer",
            "long_candidate_wer",
            "sigma",
            "rank",
            "base_seed",
            "target_group",
            "recording_ids",
        ],
    )
    logger = WandbLogger(config, run_dir, disabled=args.disable_wandb)

    clean_by_block: Dict[int, BlockWers] = {}
    scores_by_pair_sign: Dict[Tuple[int, int], List[float]] = {}

    try:
        for block in search_blocks:
            print(f"Evaluating clean search block {block.block_id}: {block.recording_ids}")
            clean_wers, clean_diagnostics = evaluate_block_for_models(
                bundles,
                block,
                use_tqdm=bool(config.get("evaluation", {}).get("use_tqdm", False)),
            )
            clean_by_block[block.block_id] = BlockWers(
                block_id=block.block_id,
                recording_ids=tuple(block.recording_ids),
                wers=clean_wers,
            )
            clean_row = {
                "phase": "search_clean",
                "block_id": block.block_id,
                "recording_ids": block.recording_ids,
                **{f"{label}_clean_wer": value for label, value in clean_wers.items()},
                "diagnostics": clean_diagnostics,
            }
            clean_writer.write(clean_row)

            for candidate in iter_candidates(num_pairs):
                print(
                    f"Evaluating block={block.block_id} candidate={candidate.candidate_id} "
                    f"pair={candidate.pair_id} sign={candidate.sign}"
                )
                damaged_wers: Dict[str, float] = {}
                damaged_diagnostics: Dict[str, Any] = {}
                for label, bundle in bundles.items():
                    with applied_candidate_perturbation(
                        bundle.model,
                        targets,
                        candidate=candidate,
                        rank=rank,
                        sigma=sigma,
                        base_seed=base_seed,
                    ):
                        result = evaluate_model_on_block(
                            bundle,
                            block,
                            use_tqdm=bool(config.get("evaluation", {}).get("use_tqdm", False)),
                        )
                    damaged_wers[label] = float(result["wer"])
                    damaged_diagnostics[label] = result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                score = score_candidate(clean_wers, damaged_wers, score_cfg)
                scores_by_pair_sign.setdefault((candidate.pair_id, candidate.sign), []).append(
                    float(score["context_score"])
                )
                row = {
                    "phase": "search_candidate",
                    "block_id": block.block_id,
                    "recording_ids": block.recording_ids,
                    "candidate_id": candidate.candidate_id,
                    "pair_id": candidate.pair_id,
                    "sign": candidate.sign,
                    "sigma": sigma,
                    "rank": rank,
                    "base_seed": base_seed,
                    "target_group": "ff1_ff2_conv",
                    **score,
                    **{f"{label}_clean_wer": clean_wers[label] for label in ("short", "medium", "long")},
                    **{
                        f"{label}_candidate_wer": damaged_wers[label]
                        for label in ("short", "medium", "long")
                    },
                    "diagnostics": damaged_diagnostics,
                }
                missing_wandb = WANDB_REQUIRED_KEYS - row.keys()
                if missing_wandb:
                    raise RuntimeError(f"Internal logging row missing W&B keys: {sorted(missing_wandb)}")
                candidate_writer.write(row)
                candidate_csv.write({**row, "recording_ids": ",".join(block.recording_ids)})
                logger.log(row)

        pair_weights: Dict[int, float] = {}
        pair_rows: List[Dict[str, Any]] = []
        for pair_id in range(num_pairs):
            pos_scores = scores_by_pair_sign.get((pair_id, 1), [])
            neg_scores = scores_by_pair_sign.get((pair_id, -1), [])
            if not pos_scores or not neg_scores:
                raise ValueError(f"Missing scores for antithetic pair {pair_id}")
            pos_mean = sum(pos_scores) / len(pos_scores)
            neg_mean = sum(neg_scores) / len(neg_scores)
            pair_weight = pos_mean - neg_mean
            pair_weights[pair_id] = pair_weight
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "positive_score_mean": pos_mean,
                    "negative_score_mean": neg_mean,
                    "pair_weight": pair_weight,
                    "num_blocks": len(pos_scores),
                }
            )
        write_json(run_dir / "pair_weights.json", pair_rows)

        validation_results: List[Dict[str, Any]] = []
        if not args.skip_validation:
            for block in validation_blocks:
                print(f"Evaluating clean validation block {block.block_id}: {block.recording_ids}")
                clean_wers, clean_diagnostics = evaluate_block_for_models(
                    bundles,
                    block,
                    use_tqdm=bool(config.get("evaluation", {}).get("use_tqdm", False)),
                )
                damaged_wers: Dict[str, float] = {}
                damaged_diagnostics: Dict[str, Any] = {}
                for label, bundle in bundles.items():
                    with applied_combined_perturbation(
                        bundle.model,
                        targets,
                        pair_weights,
                        rank=rank,
                        eta=eta,
                        base_seed=base_seed,
                    ):
                        result = evaluate_model_on_block(
                            bundle,
                            block,
                            use_tqdm=bool(config.get("evaluation", {}).get("use_tqdm", False)),
                        )
                    damaged_wers[label] = float(result["wer"])
                    damaged_diagnostics[label] = result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                score = score_candidate(clean_wers, damaged_wers, score_cfg)
                row = {
                    "phase": "validation_combined",
                    "block_id": block.block_id,
                    "recording_ids": block.recording_ids,
                    "eta": eta,
                    "sigma": sigma,
                    "rank": rank,
                    "base_seed": base_seed,
                    "target_group": "ff1_ff2_conv",
                    **score,
                    **{f"{label}_clean_wer": clean_wers[label] for label in ("short", "medium", "long")},
                    **{
                        f"{label}_candidate_wer": damaged_wers[label]
                        for label in ("short", "medium", "long")
                    },
                    "clean_diagnostics": clean_diagnostics,
                    "damaged_diagnostics": damaged_diagnostics,
                }
                validation_results.append(row)
                validation_writer.write(row)
                logger.log(row)

        if bool(search_cfg.get("save_combined_checkpoints", False)):
            checkpoint_dir = run_dir / "damaged_checkpoints"
            for label, bundle in bundles.items():
                with applied_combined_perturbation(
                    bundle.model,
                    targets,
                    pair_weights,
                    rank=rank,
                    eta=eta,
                    base_seed=base_seed,
                ):
                    save_combined_checkpoint(
                        bundle,
                        bundle.spec.path,
                        checkpoint_dir / f"{label}_combined_eta{eta:g}.pt",
                        {
                            "pair_weights": pair_rows,
                            "eta": eta,
                            "sigma": sigma,
                            "rank": rank,
                            "base_seed": base_seed,
                        },
                    )

        summary = summarise_results(clean_by_block, pair_rows, validation_results)
        write_json(run_dir / "summary.json", summary)
        write_artifact_index(
            run_dir / "ARTIFACT_INDEX.md",
            run_dir=run_dir,
            config_path=config_path,
            kept_artifacts=[
                run_dir / "config_resolved.json",
                run_dir / "target_tensors.json",
                run_dir / "blocks.json",
                run_dir / "candidate_metrics.csv",
                run_dir / "candidate_metrics.jsonl",
                run_dir / "pair_weights.json",
                run_dir / "validation_metrics.jsonl",
                run_dir / "summary.json",
            ],
            notes=[
                "Mimas-local Earnings train split is used when configured with /store/store4/data/earnings-22.",
                "The source checkpoints are loaded read-only; perturbations are applied in memory and restored.",
            ],
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        logger.finish()

    return 0


def summarise_results(
    clean_by_block: Mapping[int, BlockWers],
    pair_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    best_pair = None
    if pair_rows:
        best_pair = max(pair_rows, key=lambda row: float(row["pair_weight"]))

    validation_summary: Dict[str, Any] = {}
    if validation_rows:
        mean_context = sum(float(row["context_score"]) for row in validation_rows) / len(validation_rows)
        mean_retained = sum(float(row["retained_gain"]) for row in validation_rows) / len(validation_rows)
        mean_damage = sum(float(row["mean_damage"]) for row in validation_rows) / len(validation_rows)
        validation_summary = {
            "num_blocks": len(validation_rows),
            "mean_context_score": mean_context,
            "mean_retained_gain": mean_retained,
            "mean_damage": mean_damage,
            "mean_short_clean_wer": sum(float(row["short_clean_wer"]) for row in validation_rows)
            / len(validation_rows),
            "mean_medium_clean_wer": sum(float(row["medium_clean_wer"]) for row in validation_rows)
            / len(validation_rows),
            "mean_long_clean_wer": sum(float(row["long_clean_wer"]) for row in validation_rows)
            / len(validation_rows),
            "mean_short_perturbed_wer": sum(float(row["short_candidate_wer"]) for row in validation_rows)
            / len(validation_rows),
            "mean_medium_perturbed_wer": sum(float(row["medium_candidate_wer"]) for row in validation_rows)
            / len(validation_rows),
            "mean_long_perturbed_wer": sum(float(row["long_candidate_wer"]) for row in validation_rows)
            / len(validation_rows),
        }

    return {
        "search_blocks": len(clean_by_block),
        "best_pair": best_pair,
        "validation": validation_summary,
        "acceptable_direction_found": bool(
            validation_summary
            and validation_summary["mean_retained_gain"] > 0
            and validation_summary["mean_long_perturbed_wer"] < validation_summary["mean_short_perturbed_wer"]
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--earnings-root", default=None)
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--devices", default=None, help="Comma-separated device list, e.g. cuda:0,cuda:1")
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--max-search-blocks", type=int, default=None)
    parser.add_argument("--max-validation-blocks", type=int, default=None)
    parser.add_argument("--list-targets", action="store_true")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--run-after-listing", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--save-combined-checkpoints", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = read_config(args.config)
    config = apply_overrides(config, args)
    return run_search(config, args.config, args)


if __name__ == "__main__":
    raise SystemExit(main())
