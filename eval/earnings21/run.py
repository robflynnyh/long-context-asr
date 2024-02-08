import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()

TEST_PATH = '/mnt/parscratch/users/acp21rjf/earnings21/media'
ALL_TEXT_PATH = '/mnt/parscratch/users/acp21rjf/earnings21/transcripts/nlp_references'


def load_and_parse_test(path):
    words = []
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip().split("|")[0])
    return " ".join(words)


def fetch_data(audio_path:str = TEST_PATH, txt_path:str = ALL_TEXT_PATH):

    audio_files = [{
        'meeting': el.replace('.mp3', ''),
        'path': os.path.join(audio_path, el)
        } for el in os.listdir(audio_path) if el.endswith('.mp3')]

    text_files = [{
        'meeting': meeting,
        'path': os.path.join(txt_path, meeting+'.nlp'),
        'text': load_and_parse_test(os.path.join(txt_path, meeting+'.nlp'))
        } for meeting in [el['meeting'] for el in audio_files]]

    return audio_files, text_files



def preprocess_transcript(text:str):
    text = text.lower()
    text = text.replace('<silence>', '')
    text = text.replace('<inaudible>', '')
    text = text.replace('<laugh>', '')
    text = text.replace('<noise>', '')
    text = text.replace('<affirmative>', '')
    text = text.replace('<crosstalk>', '')    
    text = text.replace('…', '')
    text = text.replace(',', '')
    text = text.replace('-', ' ')
    text = text.replace('.', '')
    text = text.replace('?', '')
    text = re.sub(' +', ' ', text)
    return normalize(text).lower()


def main(args):
    assert args.split in ['test'], 'Split must be test!'
    data_path = TEST_PATH 
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = checkpoint['config']
    args.config = model_config
    

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class(config=args.config, args=args))
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

    audio_files, text_files = fetch_data(audio_path=data_path, txt_path=ALL_TEXT_PATH)
    meetings_keys = [el['meeting'] for el in audio_files]
    
    all_texts = []
    all_golds = []
    for rec in tqdm(range(len(meetings_keys)), total=len(audio_files)):

        print(f'Processing {rec+1}/{len(audio_files)}')
        cur_meetings = meetings_keys[rec]
        cur_audio = audio_files[rec]['path']
        
        
        cur_text = preprocess_transcript(text_files[rec]['text'])
        assert cur_meetings == text_files[rec]['meeting'] and audio_files[rec]['meeting'] == text_files[rec]['meeting'], \
            f'Meeting names do not match: {cur_meetings}, {text_files[rec]["meeting"]}, {audio_files[rec]["meeting"]}'

        audio_spec = processing_chain(cur_audio)
        print('\n-------\n'+cur_meetings+'\n-------\n')
        
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
    parser.add_argument('-beams', '--beam_width', type=int, default=1, help='beam width for decoding')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')
    parser.add_argument('-log', '--log', type=str, default='')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')
    #parser.add_argument('-pad_to', '--pad_to', default=0, type=int, help='pad sequence to pad_to')

    args = parser.parse_args()
    main(args)
    
