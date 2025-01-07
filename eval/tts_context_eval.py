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

from earnings22_full.run import get_text_and_audio as get_text_and_audio_earnings22_full
from earnings22.run import get_text_and_audio as get_text_and_audio_earnings22
from tedlium.run import get_text_and_audio as get_text_and_audio_tedlium
from rev16.run import get_text_and_audio as get_text_and_audio_rev16
from this_american_life.run import get_text_and_audio as get_text_and_audio_this_american_life
from spotify.run import get_text_and_audio as get_text_and_audio_spotify

from TTS.api import TTS

datasets_functions = {
    'earnings22_full': get_text_and_audio_earnings22_full,
    'earnings22': get_text_and_audio_earnings22,
    'tedlium': get_text_and_audio_tedlium,
    'rev16': get_text_and_audio_rev16,
    'this_american_life': get_text_and_audio_this_american_life,
    'spotify': get_text_and_audio_spotify
}


def match_gains_tensor(source_tensor, target_tensor):
    """
    Adjust the gain of the source tensor to match the target tensor's amplitude level.
    
    Parameters:
    source_tensor (torch.Tensor): Audio tensor that needs gain adjustment (shape: [channels, samples])
    target_tensor (torch.Tensor): Reference audio tensor (shape: [channels, samples])
    
    Returns:
    torch.Tensor: Gain-adjusted audio tensor
    float: The gain adjustment factor applied
    """
    
    # Calculate RMS energy
    source_rms = torch.sqrt(torch.mean(source_tensor**2))
    target_rms = torch.sqrt(torch.mean(target_tensor**2))
    
    # Calculate gain adjustment factor
    gain_factor = target_rms / source_rms
    
    # Apply gain adjustment
    adjusted_tensor = source_tensor * gain_factor
    
    # Prevent clipping
    max_amplitude = torch.max(torch.abs(adjusted_tensor))
    if max_amplitude > 1.0:
        adjusted_tensor = adjusted_tensor / max_amplitude
    
    return adjusted_tensor, gain_factor.item()


def create_mixed_recording(
        real_index, 
        synthetic_wavs, 
        segments, 
        real_audio, 
        downsample_factor,
        buffer_size=1,
        target_size=2,
    ):
    target_index=real_index
    real_start_index, real_end_index = max(target_index-buffer_size, 0),  min(target_index+buffer_size+1, len(segments)-1)    
    
    real_start_audio_idx, real_end_audio_idx = segments[real_start_index].start * downsample_factor, segments[real_end_index].end * downsample_factor
    real_duration_audio_idx = real_end_audio_idx - real_start_audio_idx
    real_prebufferduration_audio_idx = (segments[target_index].start - segments[real_start_index].start) * downsample_factor
    real_postbufferduration_audio_idx = (segments[real_end_index].end - segments[target_index + target_size - 1].end) * downsample_factor
    
    real_audio_segment = real_audio[:, int(round(real_start_audio_idx)):int(round(real_end_audio_idx))]
    synthetic_wavs_tensor = torch.cat([synthetic_wavs[i] for i in range(len(synthetic_wavs))], dim=-1)
    real_audio_segment, _ = match_gains_tensor(real_audio_segment, synthetic_wavs_tensor) # match gains so that synthetic audio is not louder than real audio

    new_segs = []
    real_audio_idx = None
    for i, seg in enumerate(segments):
        if i < real_start_index or i > real_end_index:
            new_segs.append(synthetic_wavs[i])
        elif i == target_index:
            real_audio_idx = len(new_segs)
            new_segs.append(real_audio_segment)
    frames_before_real_audio = sum([seg.shape[-1] for i,seg in enumerate(new_segs) if i < real_audio_idx])
    frames_after_real_audio = sum([seg.shape[-1] for i,seg in enumerate(new_segs) if i > real_audio_idx])
    
    frames_before_target_audio = frames_before_real_audio + real_prebufferduration_audio_idx
    frames_after_target_audio = frames_after_real_audio + real_postbufferduration_audio_idx
    real_target_duration_audio_idx = real_duration_audio_idx - real_prebufferduration_audio_idx - real_postbufferduration_audio_idx


    mixed_audio = torch.cat(new_segs, dim=-1)

    return {
        'mixed_audio': mixed_audio,
        'frames_before_target_audio': frames_before_target_audio,
        'frames_after_target_audio': frames_after_target_audio,
        'real_target_duration_audio_idx': real_target_duration_audio_idx,
        'prebufferduration_audio_idx': real_prebufferduration_audio_idx,
        'postbufferduration_audio_idx': real_postbufferduration_audio_idx,
    }



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
    synthesize_fn = partial(tts.tts, speaker="Ana Florence", language="en", speed=2.0)

    def synthesize(text):
        out = torch.as_tensor(synthesize_fn([text]))[None]
        out = resample(out, tts.synthesizer.output_sample_rate, 16000)
        return out

    # for idx, module in enumerate([el.attend.fn for el in model.layers]):
    #     module.return_attention_weights = True

    all_texts = []
    all_golds = []
    wer_data = []
    
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

        segments = segments#[:20]
     
        synthetic_wavs = {}
        for i, seg in enumerate(segments):
            synthetic_wavs[i] = synthesize(seg.text)
            print(f'{i}/{len(segments)}: {synthetic_wavs[i].shape}')
        


        target_size = 2
        output_size = (1, logits.shape[0]*2, tokenizer.vocab_size() + 1)
        output_counts = torch.zeros(output_size, dtype=torch.int)
        output_probs = torch.zeros(output_size, dtype=torch.float)
        if len(segments) < target_size: raise ValueError('The number of segments must be larger than the target size! decrease target size, or decrease block size when segmenting')
        #print(len(segments))
        index = 0
        while index < len(segments):
            print(f'Transcribing {index+1}/{len(segments)}')
            target_size = 2 if index + 1 < len(segments) else 1
            #print('TARGET SIZE', target_size)
            mixed_audio_data = create_mixed_recording(
                real_index=index, 
                synthetic_wavs=synthetic_wavs, 
                segments=segments, 
                real_audio=audio_wav, 
                downsample_factor=downsample_factor,
                buffer_size=args.buffer_size,
                target_size=target_size,
            )
            mixed_audio = mixed_audio_data['mixed_audio']

            mixed_audio_spec = to_spectogram(mixed_audio, global_normalisation=True)
            mixed_logits = eval_fn(
                args = args, 
                model = model, 
                spec = mixed_audio_spec,
                seq_len = args.seq_len,
                overlap = args.overlap,
                tokenizer = tokenizer
            )

            downsample_factor = mixed_audio.shape[-1] / mixed_logits.shape[0]
            frames_before_target = round(mixed_audio_data['frames_before_target_audio'] / downsample_factor)
            duration = round(mixed_audio_data['real_target_duration_audio_idx'] / downsample_factor)

            mixed_probs = torch.as_tensor(mixed_logits).exp()[None]
       
            mixed_probs = mixed_probs[:, frames_before_target:frames_before_target+duration, :]
            mixed_probs_duration = mixed_probs.shape[1]
            print(output_probs[:, frames_before_target:frames_before_target+duration, :].shape, 
                  output_probs[:, frames_before_target:frames_before_target+mixed_probs_duration, :].shape,
                  mixed_probs.shape,
                  output_probs.shape,
                  frames_before_target,
                  duration,
                  mixed_probs_duration
                  )
            output_counts[:, frames_before_target:frames_before_target+mixed_probs_duration, :] += 1
            output_probs[:, frames_before_target:frames_before_target+mixed_probs_duration, :] += mixed_probs

            index += target_size


        B,N,C = output_probs.shape
        output_probs = output_probs[output_counts.sum(dim=-1) != 0]
        output_probs = output_probs.reshape(B,-1,C)
        output_counts = output_counts[output_counts.sum(dim=-1) != 0]
        output_counts = output_counts.reshape(B,-1,C)
        #print(output_counts.to(torch.float).mean(-1).squeeze(0).tolist())
        logits = output_probs / output_counts
        logits = torch.log(logits).squeeze(0) # convert to log 

        #print(logits.shape)

        out_text = decoder(logits)

        out = normalize(out_text).lower()
        
        if verbose: print(gold_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(gold_text)
        wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
        print(f'WER: {wer}')   
        exit()

        if include_per_recording_evaluations:
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
            wer_data.append({
                'recording': data[rec]['id'],
                'wer': wer,
                'words': words,
                'ins_rate': ins_rate,
                'del_rate': del_rate,
                'sub_rate': sub_rate
            })

        if args.__dict__.get('break_eval', False): break

        

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)

    if verbose: print(f'WER: {wer}')

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
    parser.add_argument('-eval_mode', '--evaluation_mode', type=str, default='windowed_attention', choices=['averaged_moving_window', 'windowed_attention', 'buffered'])
    parser.add_argument('-buffer_size', '--buffer_size', type=int, default=1, help='buffer size for buffered evaluation')

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