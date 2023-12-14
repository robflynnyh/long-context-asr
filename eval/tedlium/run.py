import torch, lcasr, os, re
import argparse
from tqdm import tqdm
from typing import List, Tuple
from lcasr.utils.audio_tools import processing_chain, total_seconds, total_frames
from lcasr.utils.general import load_model, get_model_class
from lcasr.eval.utils import zero_out_spectogram, fetch_logits, decode_beams_lm
from lcasr.eval.wer import word_error_rate_detail 
from pyctcdecode import build_ctcdecoder

TEST_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/test/'
DEV_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/dev/'

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
    for line in stm:
        sline = line.split(' ')
        if len(sline) < 6:
            continue
        a_id, s_id, spk, start, end, meta = sline[:6]
        text = ' '.join(sline[6:])
        if text == 'ignore_time_segment_in_scoring':
            continue
        utterances.append({
            'start': float(start), 
            'end': float(end), 
            'text': text, 
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




def main(args):
    assert args.split in ['test', 'dev'], f'Split must be either test or dev (got {args.split})'
    data_path = TEST_PATH if args.split == 'test' else DEV_PATH

    
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
    model.device = device
    model = model.to(device)
    model.eval()

    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)


    audio_files, text_files = fetch_data(path=data_path)
    paired = dict(zip(audio_files, text_files))
    
    all_texts = []
    all_golds = []

    if not args.single_utterance:
        for rec in tqdm(range(len(audio_files)), total=len(audio_files)):
            print(f'Processing {rec+1}/{len(audio_files)}') if args.verbose else None   

            audio_spec = processing_chain(audio_files[rec])
            print('\n\n'+paired[audio_files[rec]]+'\n\n') if args.verbose else None
            stm_path = paired[audio_files[rec]]
            gold_text, timings, remove_timings = proc_stm_and_timings(stm_path=stm_path)

            audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings, buffer=-0.5)
            import time
            stime = time.time()
            logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
            etime = time.time()
            print(f'Inference time: {etime-stime}')
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
           
            out_texts = []
            for utterance in tqdm(utterances):
                logit = fetch_logits(args, model, utterance['spectogram'], utterance['spectogram'].shape[-1], 0, tokenizer, use_tqdm=False)
                ds_factor = utterance['spectogram'].shape[-1] / logit.shape[0]
                decoded, bo = decode_beams_lm([logit], decoder, beam_width=1, ds_factor=ds_factor)
                out_text = normalize(decoded[0]['text']).lower().strip()
                out_text = out_text[:-1].strip() if out_text.endswith('.') else out_text
                out_texts.append(out_text)
            all_text = " ".join(out_texts).strip()#
            gold_text = normalize(gold_text).lower().strip()
            print(gold_text) if args.verbose else None
            print(all_text) if args.verbose else None
            all_texts.append(all_text)
            all_golds.append(gold_text)
            

        
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')

    if args.log != '':
        with open(args.log, 'a') as f:
            f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {wer}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')

    parser.add_argument('-single_utt', '--single_utterance', action='store_true', help='single utterance decoding')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')

    parser.add_argument('-log', '--log', type=str, default='')

    args = parser.parse_args()
    args.verbose = not args.not_verbose


    main(args)
    
