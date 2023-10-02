import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim


def dynamic_eval(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=optim.AdamW,
        lr_args:dict={'lr':1e-5},
    ):

    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    optimizer = optim(model.parameters(), **lr_args)
 
    if seq_len > spec_n:
        seq_len, overlap = spec_n, 0
    else:
        overlap = overlap if overlap != -1 else args.config['audio_chunking']['overlap']
    cache_len = args.cache_len if args.cache_len != -1 else args.config['training']['max_seq_len']

    assert overlap / downsampling_factor == overlap // downsampling_factor, 'Overlap must be a multiple of the downsampling factor'
    print(f'Using seq_len: {seq_len} and overlap: {overlap} and cache_len: {cache_len}')

    all_logits, logit_count = torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1)), torch.zeros((1, spec_n//4 + seq_len, tokenizer.vocab_size() + 1))
    prev_cache, last_ulen, kill_next, logit_position = None, None, False, 0

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