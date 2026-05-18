import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch, lcasr
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.buffered_transcription import fetch_logits as buffered_eval
from lcasr.utils.general import load_model, get_model_class
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from lcasr.decoding.greedy import GreedyCTCDecoder
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
from tqdm import tqdm

from earnings22_full.run import get_text_and_audio as get_text_and_audio_earnings22_full
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16
from this_american_life.run import get_text_and_audio as get_text_and_audio_this_american_life
from spotify.run import get_text_and_audio as get_text_and_audio_spotify
from floras50.run import get_text_and_audio as get_text_and_audio_floras50

datasets_functions = {
    'earnings22_full': get_text_and_audio_earnings22_full,
    'earnings22': get_text_and_audio_earnings22,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16,
    'this_american_life': get_text_and_audio_this_american_life,
    'spotify': get_text_and_audio_spotify,
    'floras50': get_text_and_audio_floras50,
}


def get_transcribe_kwargs(args, verbose):
    transcribe_kwargs = args.__dict__.get('transcribe_kwargs', {})
    if transcribe_kwargs is None:
        transcribe_kwargs = {}
    if not hasattr(transcribe_kwargs, 'items'):
        raise TypeError('transcribe_kwargs must be a mapping')

    transcribe_kwargs = {key: value for key, value in transcribe_kwargs.items()}
    transcribe_kwargs.setdefault('verbose', verbose)
    return transcribe_kwargs


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
            all_text = result.get('text', '')
        else:
            all_text = result
        return normalize_prediction(all_text), metadata

    logits = eval_fn(
        args=args,
        model=model,
        spec=audio_spec,
        seq_len=args.seq_len,
        overlap=args.overlap,
        tokenizer=tokenizer,
    )
    out_text = decoder(torch.as_tensor(logits))
    return normalize_prediction(out_text), metadata


def write_jsonl(path, records):
    if not path:
        return
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
    report_samples = summary.get('report_samples', 5)
    for record in records[:report_samples]:
        prediction = record['prediction'].replace('|', '\\|') or '<empty>'
        reference = record['reference'].replace('|', '\\|')
        pred_ns = record.get('pred_non_silence_fraction')
        pred_ns_text = '' if pred_ns is None else f'{pred_ns:.3f}'
        lines.append(f"| `{record['recording']}` | {reference} | {prediction} | {pred_ns_text} |")
    lines.append('')
    return '\n'.join(lines)


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config

    if args.__dict__.get('disable_flash_attention', False): args.config.model.flash_attn = False

    eval_fn = moving_average_eval
    if args.__dict__.get('evaluation_mode', 'averaged_moving_window') == 'windowed_attention':
        if args.__dict__.get('window_size', None) is None:
            seq_len = args.seq_len
            subsample_factor = args.config.model.get('subsampling_factor', 8)
            ds_seq_len = seq_len // subsample_factor
            window_size = ds_seq_len // 2 # //2 because applied in both directions
        else: window_size = args.window_size
        args.config.model.attention_window_size = window_size
        args.seq_len = args.__dict__.get('max_sequence_length', 3600000) # 10 hours
    if args.__dict__.get('evaluation_mode', 'averaged_moving_window') == 'buffered': eval_fn = buffered_eval
    if args.__dict__.get('overide_window_size', None) is not None:
        args.config.model.attention_window_size = args.overide_window_size
    
    include_per_recording_evaluations = args.__dict__.get('include_per_recording_evaluations', False)

    verbose = args.__dict__.get('verbose', True)   
    transcribe_kwargs = get_transcribe_kwargs(args, verbose)

    tokenizer = {}
    if args.__dict__.get("tokenizer_path", None) is not None:
        tokenizer = {"tokenizer_path": args.tokenizer_path}
        print("Using tokenizer path from args:", args.tokenizer_path)

    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer)
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': args.config.get('model_class', args.model_class)}))
    model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_dtype = select_dtype(args.__dict__.get('eval_dtype', 'float32'), device)
    model.device = device
    model = model.to(device=device, dtype=eval_dtype) if eval_dtype is not None else model.to(device)
    model.eval()

    decoder = None
    if not hasattr(model, 'transcribe'): decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)

    data = datasets_functions[args.dataset](args.split, **get_dataset_kwargs(args))
    if args.__dict__.get('max_recordings', None) is not None:
        data = data[:args.max_recordings]

    # for idx, module in enumerate([el.attend.fn for el in model.layers]):
    #     module.return_attention_weights = True

    all_texts = []
    all_golds = []
    wer_data = []
    records = []
    utterance_count = 0
    max_utterances = args.__dict__.get('max_utterances', None)
    utterance_level = args.__dict__.get('utterance_level', False)

    pbar = tqdm(range(len(data)), total=len(data), disable=args.__dict__.get('no_progress', False)) #if verbose else range(len(data))
    for rec in pbar:
        if verbose: print(f'Processing {rec+1}/{len(data)}')

        if verbose: print('\n-------\n'+data[rec]['id']+'\n-------\n')
        processed = data[rec]['process_fn'](data[rec])
        if utterance_level:
            utterances, _ = processed
            eval_items = []
            recording_id = Path(data[rec]['id']).stem
            for utt_idx, utterance in enumerate(utterances):
                eval_items.append({
                    'recording': utterance.get('id', f'{recording_id}:{utt_idx}'),
                    'recording_id': utterance.get('recording_id', recording_id),
                    'speaker': utterance.get('speaker'),
                    'start': utterance.get('start'),
                    'end': utterance.get('end'),
                    'audio_spec': utterance['spectogram'],
                    'gold_text': normalize_prediction(utterance['text']),
                })
        else:
            audio_spec, gold_text = processed
            eval_items = [{
                'recording': data[rec]['id'],
                'recording_id': Path(data[rec]['id']).stem,
                'audio_spec': audio_spec,
                'gold_text': normalize_prediction(gold_text),
            }]

        for eval_item in eval_items:
            out, metadata = decode_audio(
                model=model,
                audio_spec=eval_item['audio_spec'],
                tokenizer=tokenizer,
                args=args,
                device=device,
                eval_fn=eval_fn,
                decoder=decoder,
                transcribe_kwargs=transcribe_kwargs,
            )

            gold_text = eval_item['gold_text']
            if verbose: print(gold_text, '\n', out, '\n\n')

            all_texts.append(out)
            all_golds.append(gold_text)
            record = {
                'recording': eval_item['recording'],
                'recording_id': eval_item['recording_id'],
                'speaker': eval_item.get('speaker'),
                'start': eval_item.get('start'),
                'end': eval_item.get('end'),
                'reference': gold_text,
                'prediction': out,
                'raw_prediction': metadata.get('text', out),
                'output_frames': metadata.get('output_frames'),
                'pred_non_silence_fraction': metadata.get('pred_non_silence_fraction'),
            }
            records.append(record)

            if include_per_recording_evaluations:
                wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
                wer_data.append({
                    'recording': eval_item['recording'],
                    'wer': wer,
                    'words': words,
                    'ins_rate': ins_rate,
                    'del_rate': del_rate,
                    'sub_rate': sub_rate
                })

            utterance_count += 1
            if max_utterances is not None and utterance_count >= max_utterances:
                break

        if max_utterances is not None and utterance_count >= max_utterances:
            break

        if args.__dict__.get('break_eval', False): break

    if not all_texts:
        raise RuntimeError('No evaluation items selected')

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    if verbose: print(f'WER: {wer}')

    pred_ns_values = [record['pred_non_silence_fraction'] for record in records if record.get('pred_non_silence_fraction') is not None]
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'split': args.split,
        'tedlium_root': args.__dict__.get('tedlium_root', None),
        'utterance_level': utterance_level,
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
    write_jsonl(args.__dict__.get('output_jsonl', ''), records)
    if args.__dict__.get('summary_json', ''):
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if args.__dict__.get('report_md', ''):
        report_path = Path(args.report_md)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            render_report(summary, records, args.__dict__.get('command', 'python run.py')),
            encoding='utf-8',
        )
    if args.__dict__.get('output_jsonl', '') or args.__dict__.get('summary_json', ''):
        print(json.dumps(summary, indent=2, sort_keys=True))

    wer_data.append({
        'recording': 'all',
        'wer': wer,
        'words': words,
        'ins_rate': ins_rate,
        'del_rate': del_rate,
        'sub_rate': sub_rate
    })
    return wer_data, model_config
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())

    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')
    parser.add_argument('-repeat', '--repeat', type=int, default=1, help='number of times to rerun evaluation')
    parser.add_argument('-eval_mode', '--evaluation_mode', type=str, default='averaged_moving_window', choices=['averaged_moving_window', 'windowed_attention', 'buffered'])
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

    parser.add_argument('-break', '--break_eval', action='store_true', help='break after first recording') 
    args = parser.parse_args()
    args.command = ' '.join(['PYTHONPATH=.', 'python', 'eval/run.py'] + __import__('sys').argv[1:])
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 
