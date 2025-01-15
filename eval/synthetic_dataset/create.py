import torch, argparse, lcasr
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.buffered_transcription import fetch_logits as buffered_eval
from lcasr.utils.audio_tools import grab_left_channel, resample, to_spectogram
from lcasr.eval.force_align import force_align
from lcasr.utils.general import load_model, get_model_class
from pyctcdecode import build_ctcdecoder
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from lcasr.utils.audio_tools import total_frames, total_seconds
from lcasr.decoding.greedy import GreedyCTCDecoder
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
from tqdm import tqdm
import torchaudio
from functools import partial
import random
import pickle as pkl
from os.path import join

import os, sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from earnings22_full.run import get_text_and_audio as get_text_and_audio_earnings22_full
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16
from this_american_life.run import get_text_and_audio as get_text_and_audio_this_american_life
from spotify.run import get_text_and_audio as get_text_and_audio_spotify
from typing import List

from TTS.api import TTS

datasets_functions = {
    'earnings22_full': get_text_and_audio_earnings22_full,
    'earnings22': get_text_and_audio_earnings22,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16,
    'this_american_life': get_text_and_audio_this_american_life,
    'spotify': get_text_and_audio_spotify
}

def create_conditioning_samples(files:List[str]):
    snippets = []
    for file in files:
        audio, sr = torchaudio.load(file)
        # select 6 10s snippets
        for i in range(6):
            start = random.randint(0, audio.shape[-1] - sr*10)
            end = start + sr*10
            snippet = audio[:, start:end]
            snippets.append(snippet)
        del audio
    return snippets

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
    
    include_per_recording_evaluations = args.__dict__.get('include_per_recording_evaluations', False)

    verbose = args.__dict__.get('verbose', True)   

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': args.config.get('model_class', args.model_class)}))
    model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)

    data = datasets_functions[args.dataset](args.split)

    def split_into_sentences(self, text): return text
    tts =  TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.synthesizer.split_into_sentences = partial(split_into_sentences, tts.synthesizer)

    def synthesize(text, speaker="Ana Florence", language="en", speed=2.0, speaker_wav=None):
        out = tts.tts([text], speaker=speaker, language=language, speed=speed, speaker_wav=speaker_wav)
        out = torch.as_tensor(out)[None]
        out = resample(out, tts.synthesizer.output_sample_rate, 16000)
        return out

    # for idx, module in enumerate([el.attend.fn for el in model.layers]):
    #     module.return_attention_weights = True

    snippet_paths = []
    if args.__dict__.get('condition_on', 'none') != 'none':
        conditioning_data = datasets_functions[args.condition_on]("test")
        audio_files = [el['audio'] for el in conditioning_data]
        tmp_dir = os.environ.get('TMPDIR', '/tmp')
        snippets = create_conditioning_samples(audio_files)
        for i, snippet in enumerate(snippets):
            path = join(tmp_dir, f'snippet_{i}_{random.randint(0, 1000000)}.wav')
            torchaudio.save(path, snippet, 16000)
            snippet_paths.append(path)
    print(f'Created {len(snippet_paths)} conditioning samples')

    
    pbar = tqdm(range(len(data)), total=len(data)) #if verbose else range(len(data))
    for rec in pbar:
        if verbose: print(f'Processing {rec+1}/{len(data)}')
        
        if verbose: print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
        #print(f'audio path: {data[rec]["audio"]}')
        audio_wav, sr = torchaudio.load(data[rec]['audio'])
        audio_wav = grab_left_channel(audio_wav)
        audio_wav = resample(audio_wav, sr, 16000)

      
        spec_length_s = total_seconds(spectogram_length=audio_spec.shape[-1])
     
        logits = eval_fn(
            args = args, 
            model = model, 
            spec = audio_spec,
            seq_len = args.seq_len,
            overlap = args.overlap,
            tokenizer = tokenizer
        ) 
        downsample_factor = audio_wav.shape[-1] / logits.shape[0]
        #print('DOWNSAMPLE FACTOR', downsample_factor)

        block_sizes_seconds = 10.24
        segments = force_align(
            logits = logits,
            transcript = data[rec]['text'],
            tokenizer = tokenizer,
            seconds_per_frame = spec_length_s / logits.shape[0],
            block_sizes_seconds = block_sizes_seconds
        )

        segments = segments
     
        synthetic_wavs = {}
        for i, seg in enumerate(segments):
            if len(snippet_paths) == 0:
                synthetic_wavs[i] = synthesize(seg.text)
            else:
                synthetic_wavs[i] = synthesize(seg.text, speaker_wav=snippet_paths, speaker=None)
            print(f'{i}/{len(segments)}: {synthetic_wavs[i].shape}')

        path = join(args.save_path, f'{data[rec]["id"]}.pkl')
        with open(path, 'wb') as f:
            pkl.dump({
                'synthetic_wavs': synthetic_wavs,
                'segments': segments,
                'real_audio': audio_wav,
                'downsample_factor': downsample_factor
            }, f)

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=datasets_functions.keys())

    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')
    parser.add_argument('-repeat', '--repeat', type=int, default=1, help='number of times to rerun evaluation')
    parser.add_argument('-eval_mode', '--evaluation_mode', type=str, default='windowed_attention', choices=['averaged_moving_window', 'windowed_attention', 'buffered'])

    parser.add_argument('-save_path', '--save_path', type=str, required=True, help='path to save data')
    parser.add_argument('--condition_on', type=str, default='none')

    parser.add_argument('-break', '--break_eval', action='store_true', help='break after first recording') 
    args = parser.parse_args()
    main(args)
    


#tts --text "good morning and welcome to the despegar third quarter ' 21 earnings conference call. a slide" --model_name "tts_models/multilingual/multi-dataset/xtts_v2" --speaker_idx "Ana Florence" --language_idx="en

# from TTS.api import TTS
# tts =  TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
# wav = tts.tts(text='hello world', speaker="Ana Florence", language="en")
# torchaudio.save('out.wav', torch.as_tensor(wav)[None], tts.synthesizer.output_sample_rate)
# def split_into_sentences(self, text): return text
# from functools import partial
# split_into_sentences = partial(split_into_sentences, tts.synthesizer)
# tts.synthesizer.split_into_sentences = split_into_sentences