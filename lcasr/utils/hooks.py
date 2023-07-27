import torch, torch.nn as nn

# backward hooks
def grad_sum(grad):
    return grad.sum()

def grad_mean(grad):
    return grad.mean()

def grad_near_zero(grad, decimals=2):
    grad = torch.round(grad, decimals=decimals)
    num_zeros = (grad == 0).sum()
    percent_zeros = num_zeros / grad.numel()
    return percent_zeros

def grad_sign(grad):
    return torch.sign(grad).sum()

def grad_max(grad):
    return grad.max()

def grad_min(grad):
    return grad.min()

def grad_std(grad):
    return grad.std()

def grad_norm(grad):
    return grad.norm()

debug_hooks = {
    #"sum": grad_sum,
    #"mean": grad_mean,
    "near_zero": grad_near_zero,
    #"sign": grad_sign,
    #"max": grad_max,
    #"min": grad_min,
    "std": grad_std,
    "norm": grad_norm
}


def add_debug_backwards_hooks(model, logger):
    for name, param in model.named_parameters():
        if param.requires_grad:
            for hook_name, hook in debug_hooks.items():
                param.register_hook(lambda grad, hook=hook, hook_name=hook_name, name=name: logger(
                    {f"{name}_{hook_name}_grad": hook(grad.detach()).item()}
                ))
                
    for name, module in model.named_modules():
        if isinstance(module, nn.SiLU) or isinstance(module, nn.ReLU) or isinstance(module, nn.GELU):
            for hook_name, hook in debug_hooks.items():
                module.register_backward_hook(lambda module, grad_input, grad_output, hook=hook, hook_name=hook_name, name=name: logger(
                    {f"{name}_{hook_name}_grad": hook(grad_output[0].detach()).item()}
                ))
                