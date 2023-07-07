import math
import torch
from torch import Tensor
import torch.nn.functional as F

# 3 modes: soft, max_prob, sum_prob, default is soft
def wctc_loss(log_probs : torch.Tensor, targets : torch.Tensor, input_lengths : torch.Tensor, target_lengths : torch.Tensor, 
    blank : int = 0, reduction : str = 'none', finfo_min_fp32: float = torch.finfo(torch.float32).min, finfo_min_fp16: float = torch.finfo(torch.float16).min, 
    alignment : bool = False, mode : str = 'soft', return_mean : bool = False, zero_infinity: bool = False):

    # move to the same device
    targets = targets.to(log_probs.device)
    input_lengths = input_lengths.to(log_probs.device).long()
    target_lengths = target_lengths.to(log_probs.device).long()

    input_time_size, batch_size = log_probs.shape[:2]
    B = torch.arange(batch_size, device = input_lengths.device)

    # handle flattened targets
    if len(targets.shape) == 1: 
        max_len = max(target_lengths)
        targets_copy = torch.full((batch_size, max_len), 0, device=log_probs.device, dtype=torch.long)
        i = 0
        cnt = 0
        for l in target_lengths:
            targets_copy[cnt, :l] = targets[i:i+l]
            i += l
            cnt += 1
    else:
        targets_copy = targets
    
    _t_a_r_g_e_t_s_ = torch.cat([targets_copy, targets_copy[:, :1]], dim = -1)
    _t_a_r_g_e_t_s_ = torch.stack([torch.full_like(_t_a_r_g_e_t_s_, blank), _t_a_r_g_e_t_s_], dim = -1).flatten(start_dim = -2)

    # make any padding token to 0
    _t_a_r_g_e_t_s_[_t_a_r_g_e_t_s_ < 0] = blank

    # make target 1 a diff_label
    diff_labels = torch.cat([torch.as_tensor([[False, True]], device = targets_copy.device).expand(batch_size, -1), _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2]], dim = 1)
    
    zero_padding, zero = 2, torch.tensor(finfo_min_fp16 if log_probs.dtype == torch.float16 else finfo_min_fp32, device = log_probs.device, dtype = log_probs.dtype)
    log_probs_ = log_probs.gather(-1, _t_a_r_g_e_t_s_.expand(input_time_size, -1, -1))
    log_alpha = torch.full((input_time_size, batch_size, zero_padding + _t_a_r_g_e_t_s_.shape[-1]), zero, device = log_probs.device, dtype = log_probs.dtype)

    # add wild-card in the first row
    log_alpha[:, :, 1] = 0.0 # log prob 1 = 0

    log_alpha[0, :, zero_padding + 0] = log_probs[0, :, blank]
    log_alpha[0, :, zero_padding + 1] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]
    for t in range(1, input_time_size):
        log_alpha[t, :, 2:] = log_probs_[t] + logadd(log_alpha[t - 1, :, 2:], log_alpha[t - 1, :, 1:-1], torch.where(diff_labels, log_alpha[t - 1, :, :-2], zero))

    # track the entire last row
    l1l2 = log_alpha.gather(-1, torch.stack([zero_padding + target_lengths * 2 - 1, zero_padding + target_lengths * 2], dim = -1).repeat(input_time_size, 1, 1))
    l1l2_sum = torch.logsumexp(l1l2, dim=-1)

    # 3 different modes
    if mode == 'soft':
        l1l2_sigma = torch.sum(F.softmax(l1l2_sum, dim=0) * l1l2_sum, dim=0)
    if mode == 'max_prob':
        l1l2_sigma = torch.max(l1l2_sum, dim=0)[0]
    if mode == 'sum_prob':
        l1l2_sigma = torch.logsumexp(l1l2_sum, dim=0)

    if return_mean:
        return torch.mean(-l1l2_sigma)
    else:
        return -l1l2_sigma


class WCTCLoss(torch.nn.CTCLoss):
    def forward(self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor) -> Tensor:
        loss = wctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank, reduction=self.reduction, zero_infinity=self.zero_infinity)
        return loss


def logadd(x0, x1, x2):
    # produces nan gradients in backward if -inf log-space zero element is used https://github.com/pytorch/pytorch/issues/31829
    x0 = x0.clone()
    x1 = x1.clone()
    x2 = x2.clone()
    return torch.logsumexp(torch.stack([x0, x1, x2]), dim = 0)
    

class LogsumexpFunction(torch.autograd.function.Function):
    @staticmethod
    def forward(self, x0, x1, x2):
        m = torch.max(torch.max(x0, x1), x2)
        m = m.masked_fill_(torch.isinf(m), 0)
        e0 = (x0 - m).exp_()
        e1 = (x1 - m).exp_()
        e2 = (x2 - m).exp_()
        e = (e0 + e1).add_(e2).clamp_(min = 1e-16)
        self.save_for_backward(e0, e1, e2, e)
        return e.log_().add_(m)

    @staticmethod
    def backward(self, grad_output):
        e0, e1, e2, e = self.saved_tensors
        g = grad_output / e
        return g * e0, g * e1, g * e2

