import torch, argparse, lcasr, os, re, json
from tqdm import tqdm
from typing import Tuple
from lcasr.utils import audio_tools 
from lcasr.eval.utils import zero_out_spectogram
from lcasr.eval.utils import fetch_logits, decode_beams_lm
from lcasr.utils.general import load_model
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
from whisper.normalizers import EnglishTextNormalizer
import torchaudio
import soundfile as sf
normalize = EnglishTextNormalizer()
import re

TEST_AUDIO = '/mnt/parscratch/users/acp21rjf/chime6/audio/eval'
DEV_AUDIO = '/mnt/parscratch/users/acp21rjf/chime6/audio/dev'
TEST_TEXT = '/mnt/parscratch/users/acp21rjf/chime6/transcriptions/eval'
DEV_TEXT = '/mnt/parscratch/users/acp21rjf/chime6/transcriptions/dev'

DATA = {
    'test': {
        'audio': TEST_AUDIO,
        'text': TEST_TEXT
    },
    'dev': {
        'audio': DEV_AUDIO,
        'text': DEV_TEXT
    }
}


def combine_and_load_audio(audio_files:list, stime:float, etime:float) -> torch.Tensor:
    '''
    There are several particpants in each meeting, and each participant has several channels and their own audio file.
    So we want to combine all the channels for each participant and then combine all the participants.
    '''
    # load all audio files
    audios = []
    for audio_file in audio_files:
        audio, _ = torchaudio.load(audio_file)
        audio = audio.mean(dim=0)
        audios.append(audio)
        print(audio.shape)
    max_len = max([audio.shape[-1] for audio in audios])
    # pad from the right
    audios = [torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1])) for audio in audios]
   
    audio = torch.stack(audios, dim=0).mean(dim=0)[None]
    spec = audio_tools.to_spectogram(waveform=audio, global_normalisation=True)
    # get duration in seconds
    spec_duration = audio_tools.total_seconds(spec.shape[-1])
 
    remove_start, remove_end = {'start':0.0, 'end':stime}, {'start':etime, 'end':spec_duration+10}
    remove_timings = [remove_start, remove_end]
    spec = zero_out_spectogram(spec, remove_timings, buffer=0.0)

    return spec


def convert_str_to_seconds(time_str:str) -> float: # in format: HH:MM:SS convert to seconds
    hours, minutes, seconds = time_str.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def fetch_data(data:dict = DATA['test']) -> Tuple[list, list]:
    # get text
    text_files = {}
    start_times = {}
    end_times = {}
    for filename in os.listdir(data['text']):
        if filename.endswith('.json'):
            with open(os.path.join(data['text'], filename), 'r') as f:
                j_data, Sname = json.load(f), filename.replace('.json', '')
                text_files[Sname] =  " ".join([el['words'] for el in j_data])
                stime, etime = list(map(convert_str_to_seconds, [j_data[0]['start_time'], j_data[-1]['end_time']]))
                start_times[Sname] = stime
                end_times[Sname] = etime
    
    # get audio
    audio_file_names = [el for el in os.listdir(data['audio']) if el.endswith('.wav')]
    audio_file_names = [el for el in audio_file_names if re.match('S\d+_P\d+.wav', el)]
    # get unique scene names
    scene_names = list(set([el.split('_')[0] for el in audio_file_names]))
    audio_files = {k:[] for k in scene_names}
    for filename in audio_file_names:
        scene_name = filename.split('_')[0]
        audio_files[scene_name].append(os.path.join(data['audio'], filename))

    # check keys match for audio and text
    assert set(audio_files.keys()) == set(text_files.keys()), 'Keys do not match'
        
    return audio_files, text_files, start_times, end_times





def main(args):
    assert args.split in ['test', 'dev'], 'Split must be either test or dev'
    data_path = DATA[args.split]
    
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

    vocab = [tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())] + [""]
    decoder = build_ctcdecoder(vocab, kenlm_model_path=None, alpha=None, beta=None)

    audio_files, text_files, stimes, etimes = fetch_data(data=data_path)
    meetings_keys = list(audio_files.keys())
    
    all_texts = []
    all_golds = []
    for rec in tqdm(range(len(meetings_keys)), total=len(audio_files)):
        cur_meeting = meetings_keys[rec]
        print(f'Processing {rec+1}/{len(audio_files)}')
        print('\n-------\n'+cur_meeting+'\n-------\n')

        cur_audio_files = audio_files[cur_meeting]
        
        cur_text = normalize(text_files[cur_meeting]).lower()
        
        audio_spec = combine_and_load_audio(cur_audio_files, stimes[cur_meeting], etimes[cur_meeting])
        
        
        
        logits = fetch_logits(args, model, audio_spec, args.seq_len, args.overlap, tokenizer)

        ds_factor = audio_spec.shape[-1] / logits.shape[0]
        decoded, bo = decode_beams_lm([logits], decoder, beam_width=args.beam_width, ds_factor=ds_factor)
        out = normalize(decoded[0]['text']).lower()
        
        print(cur_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(cur_text)
        
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
    parser.add_argument('-beams', '--beam_width', type=int, default=1, help='beam width for decoding')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')
    parser.add_argument('-log', '--log', type=str, default='')

    args = parser.parse_args()
    main(args)
    
