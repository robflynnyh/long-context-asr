import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils.audio_tools import processing_chain
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
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
    return text.lower()


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
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='none')

    
    losses = []
    target_lengths = []
    for rec in tqdm(range(len(meetings_keys)), total=len(audio_files)):
        print(f'Processing {rec+1}/{len(audio_files)}')
        cur_meetings = meetings_keys[rec]
        cur_audio = audio_files[rec]['path']
        
        
        cur_text = preprocess_transcript(text_files[rec]['text'])
        assert cur_meetings == text_files[rec]['id'] and audio_files[rec]['id'] == text_files[rec]['id'], \
            f'Meeting names do not match: {cur_meetings}, {text_files[rec]["id"]}, {audio_files[rec]["id"]}'

        audio_spec = processing_chain(cur_audio)
        print('\n-------\n'+cur_meetings+'\n-------\n')
        
        logprobs = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)
        logprobs = torch.tensor(logprobs, dtype=torch.float32, device=device).unsqueeze(0)
      

        cur_text_tokenized = tokenizer.encode(cur_text)
        cur_text_tokenized = torch.tensor(cur_text_tokenized, dtype=torch.long, device=device).unsqueeze(0)

        targets = cur_text_tokenized[cur_text_tokenized!=1][None] # remove unk  
        loss = torch.nn.CTCLoss(blank=4095, reduction='sum')(
            logprobs.transpose(0,1),
            targets,
            input_lengths = torch.LongTensor([logprobs.shape[1]]),
            target_lengths = torch.LongTensor([targets.shape[1]])
        )

        # import pickle as pkl
        # with open('logprobs.pkl', 'wb') as f:
        #     pkl.dump({
        #         'logprobs': logprobs,
        #         'cur_text_tokenized': cur_text_tokenized,
        #         'cur_text': cur_text,
        #         'tokenizer': tokenizer,
        #         'loss_fn': ctc_loss_fn,
        #         'loss': loss
        #     }, f)
        # exit()

        losses.append(loss.item())
        target_lengths.append(targets.shape[1])
        print(f'loss: {loss.item() / targets.shape[1]}')

        
    final_loss = sum(losses) / sum(target_lengths)

    print(f'loss: {final_loss}')

    if args.log != '':
        with open(args.log, 'a') as f:
            f.write(f'{args.checkpoint}\t overlap: {args.overlap}\t seq_len: {args.seq_len}\t WER: {final_loss}\n')

    return final_loss, model_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')
    parser.add_argument('-log', '--log', type=str, default='')
    #parser.add_argument('-pad_to', '--pad_to', default=0, type=int, help='pad sequence to pad_to')

    args = parser.parse_args()
    main(args)
    
