import torch


def add_eos(tokens, eos_id, token_lens):
    tokens[torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.long), (token_lens - 1).to(torch.long)] = eos_id 
    return tokens

def token_lens_to_mask(token_lens, max_len=None):
    max_len = token_lens.max() if max_len is None else max_len
    mask = torch.arange(max_len, device=token_lens.device)[None, :] < token_lens[:, None]
    return mask

def mark_padding(targets, mask, pad_id):
    targets[~mask] = pad_id
    return targets