import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import load, resample, grab_left_channel, to_spectogram
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
from audiomentations import AddGaussianSNR
import torchaudio
import random
import numpy as np

normalize = EnglishTextNormalizer()

DATA_PATH = '/mnt/parscratch/users/acp21rjf/rev_benchmark'
TEST_IDS = '/mnt/parscratch/users/acp21rjf/rev_benchmark/test.txt'


def open_txt(path:str):
    with open(path, 'r') as f:
        return f.read().strip()

def fetch_data(data_path:str = DATA_PATH, ids:str = TEST_IDS):
    with open(TEST_IDS, 'r') as f:
        IDS = f.read().strip().split(" ")
        IDS = [el.strip() for el in IDS if el.strip() != '']

    audio_files = [{
        'id': el,
        'path': os.path.join(data_path, "audio", el+".mp3"),
    } for el in IDS]

    text_files = [{
        'id': el,
        'text': open_txt(os.path.join(data_path, "transcripts", el+".txt"))
    } for el in IDS]


    return audio_files, text_files



def preprocess_transcript(text:str):
    return normalize(text).lower()

def processing_chain(
        path_in:str, 
        normalise:bool = True,
        min_snr_db=-0.0,
        max_snr_db=20.0,
        p=1.0
    ):
    waveform, sample_rate = load(path_in)
    waveform = grab_left_channel(waveform)
    waveform = resample(waveform, sample_rate, 16000)
    waveform = waveform.numpy()
    np.random.seed(17925) # seed for reproducibility
    random.seed(17925)
    transform = AddGaussianSNR(min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=p)
    augmented_waveform = transform(waveform, sample_rate=16000)
    augmented_waveform = torch.from_numpy(augmented_waveform)
    spectrogram = to_spectogram(augmented_waveform, global_normalisation=normalise)
    return spectrogram

def main(args):
    assert args.split in ['test'], 'Split must be test'
    IDS = TEST_IDS
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config
    

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size())
    tparams = model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    # if args.pad_to != 0 and hasattr(model, 'use_padded_forward'):
    #     model.use_padded_forward(pad_to = args.pad_to)

    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    audio_files, text_files = fetch_data(data_path = DATA_PATH, ids = IDS)
    meetings_keys = [el['id'] for el in audio_files]
    
    all_texts = []
    all_golds = []
    for rec in tqdm(range(len(meetings_keys)), total=len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}')
        cur_meetings = meetings_keys[rec]
        cur_audio = audio_files[rec]['path']
        
        
        cur_text = preprocess_transcript(text_files[rec]['text'])
        assert cur_meetings == text_files[rec]['id'] and audio_files[rec]['id'] == text_files[rec]['id'], \
            f'Meeting names do not match: {cur_meetings}, {text_files[rec]["id"]}, {audio_files[rec]["id"]}'
        print('\n-------\n'+cur_meetings+'\n-------\n')

        audio_spec = processing_chain(
            path_in = cur_audio,
            normalise = True,
            min_snr_db = args.min_snr_db,
            max_snr_db = args.max_snr_db,
            p = args.p
        )
        
        
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)

        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=1, ds_factor=ds_factor)
        out = normalize(decoded[0]['text']).lower()
        
        print(cur_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(cur_text)
        
    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    print(f'WER: {wer}')

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
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('--min_snr_db', type=float, default=-5.0, help='min snr for augmentation')
    parser.add_argument('--max_snr_db', type=float, default=30.0, help='max snr for augmentation')
    parser.add_argument('-p', '--p', type=float, default=1.0, help='probability of augmentation')
    #parser.add_argument('-pad_to', '--pad_to', default=0, type=int, help='pad sequence to pad_to')

    args = parser.parse_args()
    main(args)
    
