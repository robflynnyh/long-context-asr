import json
import torch, lcasr, os, re
import argparse
from tqdm import tqdm
from typing import List, Tuple
from lcasr.utils.audio_tools import processing_chain, total_seconds, total_frames
from lcasr.utils.general import load_model, get_model_class
from lcasr.eval.utils import zero_out_spectogram, fetch_logits, decode_beams_lm
from lcasr.eval.wer import word_error_rate_detail 
from pyctcdecode import build_ctcdecoder
import time
from functools import partial
from pathlib import Path

TEST_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/test/'
DEV_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/dev/'
TRAIN_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/train/'

from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

def open_stm(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    return lines

def proc_stm_and_timings(stm_path:str):
    stm = open_stm(stm_path)
    all_text = ""
    timings = []
    remove_timings = []
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            remove_timings.append({'start': float(start), 'end': float(end)})
            continue
        all_text += text + ' '
        timings.append({'start': float(start), 'end': float(end)})
    all_text = all_text.strip()
    # regex to do all of the above
    # i.e replace space followed by a apostrophe followed by a letter with just the apostrophe and letter
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    # remove multiple spaces
    all_text = re.sub(r" +", r" ", all_text)
    return all_text, timings, remove_timings

def fetch_utterances(stm_path:str, spectogram:torch.Tensor):
    stm = open_stm(stm_path)
    utterances = []
    for line_index, line in enumerate(stm):
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            continue
        utterances.append({
            'id': f'{a_id}:{line_index}',
            'recording_id': a_id,
            'speaker': spk,
            'start': float(start), 
            'end': float(end), 
            'text': re.sub(r" '([a-z])", r"'\1", text).strip(),
            'start_frame': total_frames(float(start)), 
            'end_frame': total_frames(float(end)),
            'spectogram': spectogram[:, :, total_frames(float(start)):total_frames(float(end))]
        })
    
    all_text = " ".join([el['text'] for el in utterances])
    all_text = re.sub(r" '([a-z])", r"'\1", all_text)
    all_text = re.sub(r" +", r" ", all_text)
        
    return utterances, all_text


def fetch_data(path:str = TEST_PATH):
    audio_path = os.path.join(path, 'sph')
    audio_files = [os.path.join(audio_path, el) for el in os.listdir(audio_path) if el.endswith('.sph')]
    audio_files.sort()
    text_path = os.path.join(path, 'stm')
    text_files = [os.path.join(text_path, el) for el in os.listdir(text_path) if el.endswith('.stm')]
    text_files.sort()
    assert len(audio_files) == len(text_files), 'Number of audio files and text files must match'
    return audio_files, text_files


def split_path(split: str, tedlium_root: str = None) -> str:
    if tedlium_root is not None and tedlium_root != '':
        return os.path.join(tedlium_root, split)
    if split == 'test':
        return TEST_PATH
    if split == 'dev':
        return DEV_PATH
    if split == 'train':
        return TRAIN_PATH
    raise ValueError(f'Split must be test, dev, or train (got {split})')


def select_eval_dtype(name, device):
    if name is None or name == 'float32' or device.type == 'cpu':
        return None
    if name == 'bfloat16':
        return torch.bfloat16
    if name == 'float16':
        return torch.float16
    raise ValueError(f'Unsupported eval dtype: {name}')


def get_transcribe_kwargs(model, args):
    if not hasattr(model, 'get_silence_id'):
        return {}
    return {
        'decode_mode': args.decode_mode,
        'temperature': args.temperature,
        'max_output_frames': args.max_output_frames,
        'max_tokens': args.max_tokens,
        'return_metadata': True,
    }


def unpack_transcription(result):
    if isinstance(result, dict):
        return result.get('text', ''), result
    return result, {}


def write_jsonl(path, records):
    if path == '':
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + '\n')


def process_text_and_audio_fn(rec_dict, single_utterance=False):
    audio, text = rec_dict['audio'], rec_dict['text']
    audio_spec = processing_chain(audio)

    if not single_utterance:
        gold_text, _, remove_timings = proc_stm_and_timings(stm_path=text)
        audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings)
        return audio_spec, normalize(gold_text).lower().strip()
    else:
        utterances, gold_text = fetch_utterances(stm_path=text, spectogram=audio_spec)
        return utterances, normalize(gold_text).lower().strip()



def get_text_and_audio(split, **kwargs):
    assert split in ['test', 'dev', 'train'], f'Split must be either test or dev train (got {split})'
    data_path = split_path(split, kwargs.get('tedlium_root', None))
    single_utterance = kwargs.get('single_utterance', False)
    
    audio_files, text_files = fetch_data(path=data_path)
    return_data = []
    for rec in range(len(audio_files)):
        return_data.append({
            'id': audio_files[rec],
            'text': text_files[rec], 
            'audio': audio_files[rec], 
            "process_fn": partial(process_text_and_audio_fn, single_utterance=single_utterance)
        })
    return_data = sorted(return_data, key=lambda x: x['id'])

    return return_data


def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    data_path = split_path(args.split, args.tedlium_root)

    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config
    

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class(config=args.config, args=args))
    print(f'Loaded model class: {model.__class__.__name__}')
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_dtype = select_eval_dtype(args.eval_dtype, device)
    model.device = device
    model = model.to(device=device, dtype=eval_dtype) if eval_dtype is not None else model.to(device)
    model.eval()


    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)


    audio_files, text_files = fetch_data(path=data_path)
    paired = dict(zip(audio_files, text_files))
    
    all_texts = []
    all_golds = []
    prediction_records = []

    if not args.single_utterance:
        for rec in tqdm(range(len(audio_files)), total=len(audio_files)):
            print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None   

            audio_spec = processing_chain(audio_files[rec])
            print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
            stm_path = paired[audio_files[rec]]
            gold_text, timings, remove_timings = proc_stm_and_timings(stm_path=stm_path)

            audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings, buffer=-0.5)
            
            if hasattr(model, 'transcribe'):
                all_text = model.transcribe(
                    audio_spec,
                    tokenizer,
                    device=device,
                    max_sequence_length=args.seq_len,
                    **get_transcribe_kwargs(model, args),
                )
                all_text, metadata = unpack_transcription(all_text)
                all_text = normalize(all_text).lower().strip()
                all_text = all_text[:-1].strip() if all_text.endswith('.') else all_text.strip()
                prediction_records.append({
                    'recording': os.path.basename(audio_files[rec]),
                    'reference': normalize(gold_text).lower().strip(),
                    'prediction': all_text,
                    'output_frames': metadata.get('output_frames'),
                    'pred_non_silence_fraction': metadata.get('pred_non_silence_fraction'),
                })
            else:
                logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
                ds_factor = audio_spec.shape[-1] / logits.shape[0]
                decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)

                all_text = normalize(decoded[0]['text']).lower()
                all_text = all_text[:-1].strip() if all_text.endswith('.') else all_text.strip()

            gold_text = normalize(gold_text).lower()    
            print(gold_text) if args.verbose else None
            print(all_text) if args.verbose else None
            all_texts.append(all_text)
            all_golds.append(gold_text)
            break
            
    else:
        for rec in tqdm(range(len(audio_files)), total=len(audio_files)):

            print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None

            audio_spec = processing_chain(audio_files[rec])
            print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
            stm_path = paired[audio_files[rec]]
            utterances, gold_text = fetch_utterances(stm_path=stm_path, spectogram=audio_spec)
            if args.max_utterances is not None:
                utterances = utterances[:args.max_utterances]

            if hasattr(model, 'transcribe'):
                spectrograms = [utterance['spectogram'] for utterance in utterances]
                gold_utterances = [utterance['text'] for utterance in utterances]
                out_results = model.transcribe(
                    spectrograms,
                    tokenizer,
                    device=device,
                    targets=gold_utterances,
                    **get_transcribe_kwargs(model, args),
                )
                out_texts = []
                for utterance, result in zip(utterances, out_results):
                    out_text, metadata = unpack_transcription(result)
                    out_text = normalize(out_text).lower().strip()
                    out_texts.append(out_text)
                    prediction_records.append({
                        'recording': utterance['id'],
                        'recording_id': utterance['recording_id'],
                        'speaker': utterance['speaker'],
                        'start': utterance['start'],
                        'end': utterance['end'],
                        'reference': normalize(utterance['text']).lower().strip(),
                        'prediction': out_text,
                        'output_frames': metadata.get('output_frames'),
                        'pred_non_silence_fraction': metadata.get('pred_non_silence_fraction'),
                    })
            else:
                out_texts = []
                for utterance in tqdm(utterances):
                    logit = fetch_logits(args, model, utterance['spectogram'], utterance['spectogram'].shape[-1], 0, tokenizer, use_tqdm=False)
                    ds_factor = utterance['spectogram'].shape[-1] / logit.shape[0]
                    decoded, bo = decode_beams_lm([logit], decoder, beam_width=1, ds_factor=ds_factor)
                    out_text = normalize(decoded[0]['text']).lower().strip()
                    out_text = out_text[:-1].strip() if out_text.endswith('.') else out_text
                    out_texts.append(out_text)
                    prediction_records.append({
                        'recording': utterance['id'],
                        'recording_id': utterance['recording_id'],
                        'speaker': utterance['speaker'],
                        'start': utterance['start'],
                        'end': utterance['end'],
                        'reference': normalize(utterance['text']).lower().strip(),
                        'prediction': out_text,
                    })

            all_text = " ".join(out_texts).strip()#
            gold_text = normalize(" ".join([utterance['text'] for utterance in utterances])).lower().strip()
            print(gold_text) if args.verbose else None
            print(all_text) if args.verbose else None
            all_texts.append(all_text)
            all_golds.append(gold_text)
            break


        
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')
    write_jsonl(args.output_jsonl, prediction_records)

    if args.log != '':
        with open(args.log, 'a') as f:
            f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')

    return wer, model_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')
    parser.add_argument('-pad_to', '--pad_to', default=0, type=int, help='pad sequence to pad_to')

    parser.add_argument('-single_utt', '--single_utterance', action='store_true', help='single utterance decoding')
    parser.add_argument('-ted_root', '--tedlium_root', type=str, default='', help='override TEDLIUM root directory')
    parser.add_argument('-max_utts', '--max_utterances', type=int, default=None, help='maximum TEDLIUM utterances from the first recording')
    parser.add_argument('-output_jsonl', '--output_jsonl', type=str, default='', help='optional per-utterance prediction output path')
    parser.add_argument('-eval_dtype', '--eval_dtype', choices=['float32', 'bfloat16', 'float16'], default='float32', help='model dtype for CUDA evaluation')
    parser.add_argument('-decode_mode', '--decode_mode', choices=['greedy', 'sample'], default='greedy', help='decode mode for transcribe models that expose it')
    parser.add_argument('-temperature', '--temperature', type=float, default=1.0, help='sampling temperature for transcribe models that expose it')
    parser.add_argument('-max_tokens', '--max_tokens', type=int, default=256, help='maximum decoded tokens for transcribe models that expose it')
    parser.add_argument('-max_output_frames', '--max_output_frames', type=int, default=None, help='maximum output frames for transcribe models that expose it')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')

    parser.add_argument('-log', '--log', type=str, default='')

    args = parser.parse_args()
    args.verbose = not args.not_verbose


    main(args)
    
