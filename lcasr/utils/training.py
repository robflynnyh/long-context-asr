import json
import os
from typing import Any, Dict, List, Optional

import torch

from lcasr.utils.audio_tools import load_json
from lcasr.utils.dataloading import (
    Utterance_Dataloader,
    VariableBatchSimpleDataloader,
    chunk_spectogram,
    chunk_text_json,
)


def subset_pairs(pairs: Dict[str, Dict[str, Any]], max_records: Optional[int]):
    if max_records is None:
        return pairs
    keys = sorted(pairs.keys())[:max_records]
    return {key: pairs[key] for key in keys}


def load_training_pairs(config):
    if config['data'].get('format', 'recording_manifest') == 'utterance_folder':
        return None
    pairs = load_json(config['data']['path'])
    return subset_pairs(pairs, config['data'].get('max_records', None))


def build_training_dataloader(
        args,
        tokenizer,
        seen_ids: List[str],
        random_seed: int,
        paired_data: Optional[Dict[str, Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
    ):
    batch_size = args.config['training']['batch_size'] if batch_size is None else batch_size
    if args.config['data'].get('format', 'recording_manifest') == 'utterance_folder':
        return Utterance_Dataloader(
            utterance_folder=args.config['data']['path'],
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch=args.prefetch_factor,
            seen_ids=seen_ids,
            random_seed=random_seed,
            max_records=args.config['data'].get('max_records', None),
        )

    if paired_data is None:
        paired_data = load_training_pairs(args.config)
    return VariableBatchSimpleDataloader(
        pairs=paired_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        chunk_size=args.config.audio_chunking['size'],
        chunk_overlap=args.config.audio_chunking['overlap'],
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch=args.prefetch_factor,
        seen_ids=seen_ids,
        random_seed=random_seed,
    )


def refresh_dataloader_for_epoch(args, dataloader, tokenizer, batch_size: int, seen_ids: List[str], random_seed: Any):
    """Update dataloaders in place when possible, otherwise rebuild utterance-folder loaders."""
    if hasattr(dataloader, 'update'):
        dataloader.update(
            batch_size=batch_size,
            seen_ids=seen_ids,
            random_seed=random_seed,
        )
        return dataloader

    if args.config['data'].get('format', 'recording_manifest') == 'utterance_folder':
        if random_seed == 'same':
            random_seed = args.config['training'].get('random_seed', 1234)
        return build_training_dataloader(
            args=args,
            tokenizer=tokenizer,
            seen_ids=seen_ids,
            random_seed=random_seed,
            batch_size=batch_size,
        )

    raise AttributeError(f'{type(dataloader).__name__} has no update() method')


def batch_to_chunks(batch, tokenizer, pad_id: int, chunk_size: int, chunk_overlap: int):
    if isinstance(batch, dict):
        audio = batch['audio']
        audio_lengths = batch['audio_lengths']
        ids = list(batch['ids'])
        cur_batch_size = audio.shape[0]
        return [{
            'audio': audio,
            'txt': batch['text'],
            'txt_lengths': batch['text_lengths'],
            'audio_lengths': audio_lengths,
            'selection_mask': torch.ones(cur_batch_size, dtype=torch.bool),
            'cur_culm_lengths': torch.zeros_like(audio_lengths),
        }], ids, cur_batch_size

    audio, audio_lengths, txt, ids = batch
    cur_batch_size = audio.shape[0]
    audio_chunks = chunk_spectogram(spec=audio, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    txt_chunks = [
        chunk_text_json(
            text=el,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            spectogram_length=audio.shape[-1],
        )
        for el in txt
    ]

    chunks, culm_lengths_audio = [], torch.zeros_like(audio_lengths)
    for ix, el in enumerate(audio_chunks):
        remove_mask = ~(culm_lengths_audio > audio_lengths)
        cur_chunks, cur_culm_lengths = el[remove_mask], culm_lengths_audio[remove_mask]
        cur_lengths = cur_chunks.shape[-1] - (
            cur_culm_lengths
            + cur_chunks.shape[-1]
            - audio_lengths[remove_mask]
            - chunk_overlap
        ).clamp(0)

        enc_txt_chunks = [
            torch.LongTensor(tokenizer.encode(el[ix]))
            for i, el in enumerate(txt_chunks)
            if remove_mask[i]
        ]
        enc_txt_chunks_lengths = torch.LongTensor([el.shape[0] for el in enc_txt_chunks])
        enc_txt_chunks = torch.nn.utils.rnn.pad_sequence(
            enc_txt_chunks,
            batch_first=True,
            padding_value=pad_id,
        )
        if enc_txt_chunks_lengths.max() == 0:
            continue
        chunks.append({
            'audio': cur_chunks,
            'txt': enc_txt_chunks,
            'txt_lengths': enc_txt_chunks_lengths,
            'audio_lengths': cur_lengths,
            'selection_mask': remove_mask,
            'cur_culm_lengths': cur_culm_lengths,
        })
        culm_lengths_audio[remove_mask] += cur_chunks.shape[-1] - (chunk_overlap if ix != 0 else 0)

    return chunks, ids, cur_batch_size


def prepare_diagnostics_path(config):
    """Prepare the optional JSONL metric stream used to inspect blank/deletion collapse."""
    diagnostics_path = config['training'].get('diagnostics_path', None)
    if diagnostics_path is not None:
        diagnostics_dir = os.path.dirname(diagnostics_path)
        if diagnostics_dir:
            os.makedirs(diagnostics_dir, exist_ok=True)
    return diagnostics_path


def log_training_metrics(
        wandb_config,
        diagnostics_path,
        step: int,
        epoch: int,
        loss: float,
        blank_prob: float,
        learning_rate: float,
        sequence_length: int,
        batch_size: int,
        spec_augment_active: bool,
    ):
    """Write the same training snapshot to W&B and the optional local diagnostics JSONL."""
    payload = {
        'loss': loss,
        'blank_p': blank_prob,
        'learning_rate': learning_rate,
        'sequence_length': sequence_length,
        'batch_size': batch_size,
        'epoch': epoch,
    }
    if wandb_config['use']:
        import wandb
        wandb.log({**payload, 'spec_augment': int(spec_augment_active)})

    if diagnostics_path is not None:
        with open(diagnostics_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'step': int(step),
                'epoch': int(epoch),
                'loss': float(loss),
                'blank_p': float(blank_prob),
                'learning_rate': float(learning_rate),
                'sequence_length': int(sequence_length),
                'batch_size': int(batch_size),
            }) + '\n')
