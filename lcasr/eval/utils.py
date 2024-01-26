import torch
from lcasr.utils.audio_tools import total_frames, total_seconds
from typing import List, Dict, Tuple
from tqdm import tqdm
from lcasr.models.sconformer_xl import SCConformerXL

def zero_out_spectogram(spec:torch.Tensor, remove_timings:List[Dict[str, int]], buffer:float=-0.5):
    for timing in remove_timings:
        start, end = timing['start'] - buffer, timing['end'] + buffer
        start_frame, end_frame = map(total_frames, [start, end])
        spec[:,:,start_frame:end_frame] = 0
    return spec

def decode_beams_lm(
        logits_list, 
        decoder, 
        beam_width=100, 
        encoded_lengths=None,
        ds_factor=4
    ):
    decoded_data = []
    if encoded_lengths is None:
        encoded_lengths = [len(logits) for logits in logits_list]

    def proc_text_frame(tx_frame:Tuple[str, Tuple[int, int]]):
        text, frame = tx_frame
        return {'word': text, 'start': total_seconds(frame[0] * ds_factor), 'end': total_seconds(frame[1] * ds_factor)}

    for logits, length in zip(logits_list, encoded_lengths):
 
        beams = decoder.decode_beams(
            logits = logits[:length],
            beam_width = beam_width,
        )
        decoded_data.append({
                'text': beams[0].text,
                'frames': [proc_text_frame(el) for el in beams[0].text_frames] if ds_factor != None else None,
                'ngram_score': beams[0].lm_score - beams[0].logit_score,
                'am_score': beams[0].logit_score,
                'score': beams[0].lm_score # # score = ngram_score + am_score
        })

    return decoded_data, beams[0]

@torch.no_grad() # TODO: write batched version of this!!
def fetch_logits(args, model:SCConformerXL, spec:torch.Tensor, seq_len:int, overlap:int, tokenizer, use_tqdm=True):
    spec_n = spec.shape[-1]
    downsampling_factor = model.subsampling.subsampling_factor
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']
 
    if seq_len > spec_n:
        seq_len = spec_n
        overlap = 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']
    
    #assert overlap == 0 or cache_len == 0, 'Cannot use overlap and cache_len at the same time'

    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'


    print(f'Using seq_len: {seq_len} and overlap: {overlap}')

    all_logits = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    
    logit_position = 0
    
    
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
   
        audio_chunk = audio_chunk.to(model.device)
        out = model(
            audio_signal = audio_chunk,
        )


        logits = out['final_posteriors'].detach().cpu()
        # convert to prob
        logits = torch.exp(logits)
        ds_len = logits.shape[-2]

        ratio = u_len / ds_len
        overlap_ds = int(overlap / ratio)
        if i != 0:
            logit_position -= overlap_ds

        logit_count[:, logit_position:logit_position+ds_len, :] += 1
        all_logits[:, logit_position:logit_position+ds_len, :] += logits
        logit_position += ds_len 


    B,N,C = all_logits.shape
    all_logits = all_logits[logit_count.sum(dim=-1) != 0]
    all_logits = all_logits.reshape(B,-1,C)
    logit_count = logit_count[logit_count.sum(dim=-1) != 0]
    logit_count = logit_count.reshape(B,-1,C)
    logits = all_logits / logit_count
    # convert to log 
    logits = torch.log(logits)
    
    return logits.squeeze(0).numpy()