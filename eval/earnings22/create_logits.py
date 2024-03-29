import lcasr
import argparse
from tqdm import tqdm
from typing import Tuple
from lcasr.models.sconformer_xl import SCConformerXL
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder
import os, re, json, torch

TEST_PATH = '/mnt/parscratch/users/acp21rjf/earnings22/test_original'
DEV_PATH = '/mnt/parscratch/users/acp21rjf/earnings22/dev_original'
ALL_TEXT_PATH = '/mnt/parscratch/users/acp21rjf/earnings22/full_transcripts.json'

import pickle as pkl


def fetch_data(audio_path:str = TEST_PATH, txt_path:str = ALL_TEXT_PATH):
    with open(txt_path, 'r') as f:
        all_text_json = json.load(f)

    audio_files = [{
        'meeting': el.replace('.mp3', ''),
        'path': os.path.join(audio_path, el)
        } for el in os.listdir(audio_path) if el.endswith('.mp3')]

    text_files = [{
        'meeting': el['meeting'],
        'text': all_text_json[el['meeting']]
        } for el in audio_files]
 

    return audio_files, text_files



@torch.no_grad()
def fetch_logits(args, model:SCConformerXL, spec:torch.Tensor, seq_len:int, overlap:int, tokenizer, use_tqdm=True):
    spec_n = spec.shape[-1]
    
    downsampling_factor = args.config['model']['subsampling_factor']
    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']
    if seq_len > spec_n:
        seq_len = spec_n
        overlap = 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']
    cache_len = args.cache_len if args.cache_len != -1 else args.config['training']['max_seq_len']
    #assert overlap == 0 or cache_len == 0, 'Cannot use overlap and cache_len at the same time'

    print(f'{spec_n} !!!')
    
    
    print(f'Using seq_len: {seq_len} and overlap: {overlap} and cache_len: {cache_len}')

    all_logits = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    
    logit_position = 0
    
    prev_cache = None
    last_ulen = None
    kill_next = False
    pbar = tqdm(range(0, spec_n, seq_len-overlap), total=len(range(0, spec_n, seq_len-overlap))) if use_tqdm else range(0, spec_n, seq_len-overlap)
    for i in pbar:
        audio_chunk = spec[:, :, i:i+seq_len]
        u_len = audio_chunk.shape[-1]

        if kill_next:
            break
        if last_ulen != None and u_len < last_ulen:
            kill_next = True
        last_ulen = u_len

        # if u_len < (seq_len - overlap):
        #     continue

        audio_chunk = audio_chunk.to(model.device)
        out = model(
            audio_signal = audio_chunk,
            cached_kvs = prev_cache,
            cached_kv_lengths = None if prev_cache is None else torch.LongTensor([prev_cache.shape[1]] * prev_cache.shape[0]).to(prev_cache.device)
        )

        if cache_len != 0:
            prev_cache = out['kvs_to_cache'][:, -cache_len:].clone()

        logits = out['final_posteriors'].detach().cpu()
        # convert to prob
        logits = torch.exp(logits)
        ds_len = logits.shape[-2]

        ratio = u_len / ds_len
        overlap_ds = int(overlap / ratio)
        if i != 0:
            logit_position -= overlap_ds

        #logit_position = i 
        #print(all_logits.shape, ds_len, logit_position, logit_position+ds_len, logits.shape)
      
        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 

        #print(logit_position, overlap_ds, '()', u_len, i, i+seq_len, audio_chunk.shape, ratio, ds_len)
        
        
    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    # convert to log 
    logits = torch.log(logits)
    # blank_id = logits.shape[-1]-1
    # logits = logits.argmax(dim=-1)[0].tolist()
    # print(tokenizer.decode([el for el in logits if el != blank_id]))
    
    return logits.squeeze(0).numpy()

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
    return text


def main(args):
    assert args.split in ['test', 'dev'], 'Split must be either test or dev'
    data_path = TEST_PATH if args.split == 'test' else DEV_PATH
    
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

    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""] # "" = blank
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    audio_files, text_files = fetch_data(audio_path=data_path, txt_path=ALL_TEXT_PATH)
    meetings_keys = [el['meeting'] for el in audio_files]
    
    to_save = []
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
      

        to_save.append({
            'name': cur_meetings,
            'gold': cur_text,
            'logits': logits,
            'ds_factor': ds_factor  
        })
        #break
        
        
    with open(args.save_path, 'wb') as f:
        pkl.dump(to_save, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')
    
    parser.add_argument('-s', '--save_path', type=str, default='./logits/test_logits.pkl', help='path to save logits')
    

    args = parser.parse_args()
    main(args)
    
