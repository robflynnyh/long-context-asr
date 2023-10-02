import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder


def dynamic_eval_ctc_loss(
        args, 
        model:nn.Module, 
        spec:torch.Tensor, 
        seq_len:int, 
        overlap:int, 
        tokenizer, 
        use_tqdm=True,
        optim:optim.Optimizer=optim.AdamW,
        lr_args:dict={'lr':1e-4},
    ):

    spec_n = spec.shape[-1]
    downsampling_factor = args.config['model']['subsampling_factor']
    seq_len = seq_len if seq_len != -1 else args.config['audio_chunking']['size']

    # create copy of model parameters that are not updated
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach() for p in original_model_params]

    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    optimizer = optim(model.parameters(), **lr_args)
    decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)
    augmentation = SpecAugment(
        n_time_masks = 2,
        n_freq_masks = 3,
        freq_mask_param = 42,
        time_mask_param = -1,
        min_p = 0.05,
        zero_masking = False,
    )

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
        audio_chunk = spec[:, :, i:i+seq_len] # [B, C, T]
        audio_chunk = audio_chunk.repeat(3, 1, 1) # [B, C, T]
        audio_chunk[:2] = augmentation(audio_chunk[:2]) # apply augmentation to 2 of the 3 copies

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
        pseudo_targets = decoder(out['final_posteriors'][-1].detach().cpu())
        pseudo_targets = torch.LongTensor(tokenizer.encode(pseudo_targets)).unsqueeze(0).to(model.device).repeat(2, 1)
        augmented_outs = out['final_posteriors'][:2]
        N, B = augmented_outs.shape[1], augmented_outs.shape[0]
        total_tokens_in_loss = N * B
        # calculate loss
        #print(pseudo_targets.shape, augmented_outs.shape)
        loss = ctc_loss_fn(augmented_outs.transpose(0, 1), pseudo_targets, torch.LongTensor([N] * augmented_outs.shape[0]).to(model.device), torch.LongTensor([pseudo_targets.shape[1]] * pseudo_targets.shape[0]).to(model.device)) / total_tokens_in_loss
        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(f'Loss: {loss.item()}')
      

        if cache_len != 0:
            prev_cache = out['kvs_to_cache'][:, -cache_len:].clone()

        logits = out['final_posteriors'][-1].detach().cpu()
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

    # reset model parameters
    for p, p_orig in zip(model.parameters(), original_model_params):
        p.data = p_orig.data
    
    return logits.squeeze(0).numpy()

dynamic_eval = dynamic_eval_ctc_loss