import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from whisper.normalizers import EnglishTextNormalizer

normalize = EnglishTextNormalizer()


def add_cli_args(parser):
    parser.add_argument('--utterance-level', action='store_true', help='evaluate dataset utterances when the dataset loader supports it')
    parser.add_argument('--tedlium-root', type=str, default='', help='override TEDLIUM root directory')
    parser.add_argument('--max-recordings', type=int, default=None, help='limit number of recordings loaded from the dataset')
    parser.add_argument('--max-utterances', type=int, default=None, help='limit number of utterances/items evaluated')
    parser.add_argument('--decode-mode', choices=['greedy', 'sample'], default='greedy', help='transcribe decode mode for models that support it')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature for models that support sampled decode')
    parser.add_argument('--max-output-frames', type=int, default=None, help='cap generated output frames for models that support it')
    parser.add_argument('--max-tokens', type=int, default=256, help='cap decoded output tokens for models that support it')
    parser.add_argument('--eval-dtype', choices=['float32', 'bfloat16', 'float16'], default='float32', help='dtype for model evaluation on CUDA')
    parser.add_argument('--output-jsonl', type=str, default='', help='optional path for per-item predictions')
    parser.add_argument('--summary-json', type=str, default='', help='optional path for aggregate summary JSON')
    parser.add_argument('--report-md', type=str, default='', help='optional path for compact Markdown report')
    parser.add_argument('--report-samples', type=int, default=5, help='number of prediction rows to include in the Markdown report')
    parser.add_argument('--no-progress', action='store_true', help='disable tqdm progress display')


def command_from_argv(argv):
    return ' '.join(['PYTHONPATH=.', 'python', 'eval/run.py'] + argv[1:])


def get_dataset_kwargs(args):
    dataset_kwargs = args.__dict__.get('dataset_kwargs', {}) or {}
    if not hasattr(dataset_kwargs, 'items'):
        raise TypeError('dataset_kwargs must be a mapping')
    dataset_kwargs = {key: value for key, value in dataset_kwargs.items()}
    if args.__dict__.get('utterance_level', False):
        dataset_kwargs.setdefault('single_utterance', True)
    tedlium_root = args.__dict__.get('tedlium_root', None)
    if tedlium_root:
        dataset_kwargs['tedlium_root'] = tedlium_root
    return dataset_kwargs


def select_dtype(name, device):
    if name is None or name == 'float32' or device.type == 'cpu':
        return None
    if name == 'bfloat16':
        return torch.bfloat16
    if name == 'float16':
        return torch.float16
    raise ValueError(f'Unsupported eval_dtype: {name}')


def normalize_prediction(text):
    text = normalize(text).lower().strip()
    text = text[:-1].strip() if text.endswith('.') else text
    return re.sub(r' +', ' ', text)


def iter_eval_items(recording, processed, utterance_level):
    if not utterance_level:
        audio_spec, gold_text = processed
        yield {
            'recording': recording['id'],
            'recording_id': Path(recording['id']).stem,
            'audio_spec': audio_spec,
            'gold_text': gold_text,
        }
        return

    utterances, _ = processed
    recording_id = Path(recording['id']).stem
    for utt_idx, utterance in enumerate(utterances):
        yield {
            'recording': utterance.get('id', f'{recording_id}:{utt_idx}'),
            'recording_id': utterance.get('recording_id', recording_id),
            'speaker': utterance.get('speaker'),
            'start': utterance.get('start'),
            'end': utterance.get('end'),
            'audio_spec': utterance['spectogram'],
            'gold_text': normalize_prediction(utterance['text']),
        }


def decode_audio(model, audio_spec, tokenizer, args, device, eval_fn, decoder, transcribe_kwargs):
    metadata = {}
    if hasattr(model, 'transcribe'):
        kwargs = {key: value for key, value in transcribe_kwargs.items()}
        if hasattr(model, 'get_silence_id'):
            kwargs.setdefault('decode_mode', args.__dict__.get('decode_mode', 'greedy'))
            kwargs.setdefault('temperature', args.__dict__.get('temperature', 1.0))
            kwargs.setdefault('max_output_frames', args.__dict__.get('max_output_frames', None))
            kwargs.setdefault('max_tokens', args.__dict__.get('max_tokens', 256))
            kwargs.setdefault('return_metadata', True)
        result = model.transcribe(
            audio_spec,
            tokenizer,
            device=device,
            max_sequence_length=args.seq_len,
            **kwargs,
        )
        if isinstance(result, dict):
            metadata = result
            text = result.get('text', '')
        else:
            text = result
        return normalize_prediction(text), metadata

    logits = eval_fn(
        args=args,
        model=model,
        spec=audio_spec,
        seq_len=args.seq_len,
        overlap=args.overlap,
        tokenizer=tokenizer,
    )
    return normalize_prediction(decoder(torch.as_tensor(logits))), metadata


def prediction_record(eval_item, prediction, metadata):
    return {
        'recording': eval_item['recording'],
        'recording_id': eval_item['recording_id'],
        'speaker': eval_item.get('speaker'),
        'start': eval_item.get('start'),
        'end': eval_item.get('end'),
        'reference': eval_item['gold_text'],
        'prediction': prediction,
        'raw_prediction': metadata.get('text', prediction),
        'output_frames': metadata.get('output_frames'),
        'pred_non_silence_fraction': metadata.get('pred_non_silence_fraction'),
    }


def build_summary(args, records, wer_result, device, eval_dtype):
    wer, words, ins_rate, del_rate, sub_rate = wer_result
    pred_ns_values = [record['pred_non_silence_fraction'] for record in records if record.get('pred_non_silence_fraction') is not None]
    return {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'split': args.split,
        'tedlium_root': args.__dict__.get('tedlium_root', None),
        'utterance_level': args.__dict__.get('utterance_level', False),
        'decode_mode': args.__dict__.get('decode_mode', None),
        'device': str(device),
        'eval_dtype': args.__dict__.get('eval_dtype', 'float32') if eval_dtype is not None else 'float32',
        'utterances': len(records),
        'wer': wer,
        'words': words,
        'ins_rate': ins_rate,
        'del_rate': del_rate,
        'sub_rate': sub_rate,
        'mean_pred_non_silence_fraction': None if not pred_ns_values else sum(pred_ns_values) / len(pred_ns_values),
        'output_jsonl': args.__dict__.get('output_jsonl', ''),
        'summary_json': args.__dict__.get('summary_json', ''),
        'report_samples': args.__dict__.get('report_samples', 5),
    }


def write_artifacts(args, summary, records):
    output_jsonl = args.__dict__.get('output_jsonl', '')
    summary_json = args.__dict__.get('summary_json', '')
    report_md = args.__dict__.get('report_md', '')
    if output_jsonl:
        write_jsonl(output_jsonl, records)
    if summary_json:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if report_md:
        report_path = Path(report_md)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(summary, records, args.__dict__.get('command', 'python run.py')), encoding='utf-8')
    if output_jsonl or summary_json:
        print(json.dumps(summary, indent=2, sort_keys=True))


def write_jsonl(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + '\n')


def render_report(summary, records, command):
    lines = [
        '# ROB-90 Streaming Decoder TEDLIUM Utterance Eval',
        '',
        'This is a bounded wiring sanity check for the ROB-76 decoder-only streaming ASR checkpoint. Poor recognition quality is expected; the goal is to prove TEDLIUM utterance decoding runs through the generic eval path and to inspect representative outputs.',
        '',
        '## Command',
        '',
        '```bash',
        command,
        '```',
        '',
        '## Configuration',
        '',
        f"- checkpoint: `{summary['checkpoint']}`",
        f"- dataset/split: `{summary['dataset']}` / `{summary['split']}`",
        f"- TEDLIUM root: `{summary.get('tedlium_root')}`",
        f"- utterance level: `{summary['utterance_level']}`",
        f"- decode mode: `{summary.get('decode_mode')}`",
        f"- utterances: `{summary['utterances']}`",
        f"- output JSONL: `{summary.get('output_jsonl')}`",
        f"- summary JSON: `{summary.get('summary_json')}`",
        f"- device/dtype: `{summary['device']}` / `{summary['eval_dtype']}`",
        '',
        '## Summary',
        '',
        f"- WER: `{summary['wer']:.6f}`",
        f"- words: `{summary['words']}`",
        f"- insertions/deletions/substitutions: `{summary['ins_rate']:.6f}` / `{summary['del_rate']:.6f}` / `{summary['sub_rate']:.6f}`",
    ]
    if summary.get('mean_pred_non_silence_fraction') is not None:
        lines.append(f"- mean predicted non-silence fraction: `{summary['mean_pred_non_silence_fraction']:.6f}`")
    lines.extend([
        '',
        '## Sample Outputs',
        '',
        '| utterance | reference | prediction | pred non-silence |',
        '| --- | --- | --- | --- |',
    ])
    for record in records[:summary.get('report_samples', 5)]:
        prediction = record['prediction'].replace('|', '\\|') or '<empty>'
        reference = record['reference'].replace('|', '\\|')
        pred_ns = record.get('pred_non_silence_fraction')
        pred_ns_text = '' if pred_ns is None else f'{pred_ns:.3f}'
        lines.append(f"| `{record['recording']}` | {reference} | {prediction} | {pred_ns_text} |")
    lines.append('')
    return '\n'.join(lines)
