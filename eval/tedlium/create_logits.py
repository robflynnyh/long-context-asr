import torch, lcasr, os, re
import argparse
from tqdm import tqdm
from typing import List
from lcasr.utils.audio_tools import processing_chain
from lcasr.utils.general import load_model
from lcasr.eval.utils import zero_out_spectogram, fetch_logits

TEST_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/test/'
DEV_PATH = '/mnt/parscratch/users/acp21rjf/TEDLIUM_release1/dev/'

import pickle as pkl


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



    audio_files, text_files = fetch_data(path=data_path)
    paired = dict(zip(audio_files, text_files))
    
    to_save = []
    for rec in tqdm(range(len(audio_files)), total=len(audio_files)):
        #if rec < 5: continue
        print(f'Processing {rec+1}/{len(audio_files)}')

        stm_path = paired[audio_files[rec]]
        gold_text, timings, remove_timings = proc_stm_and_timings(stm_path=stm_path)

        audio_spec = processing_chain(audio_files[rec])
        audio_spec = zero_out_spectogram(spec = audio_spec, remove_timings = remove_timings)
        
        print('\n\n'+paired[audio_files[rec]]+'\n\n')
        
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)

        ds_factor = audio_spec.shape[-1] / logits.shape[0]
      

        to_save.append({
            'name': paired[audio_files[rec]],
            'gold': gold_text,
            'timings': timings,
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
    
