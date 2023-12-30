import torch, torch.nn as nn
from apex.normalization import FusedRMSNorm, FusedLayerNorm
import warnings

# base class for all models
class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def print_total_params(self, only_trainable = False):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad) if only_trainable else sum(p.numel() for p in self.parameters())
        pstr = 'Total trainable params: ' if only_trainable else 'Total params: '
        print(f'{pstr}: ', total/1e6, 'M')
        return total

    def get_param_groups(self, optim_args):
        had_blacklist = hasattr(self, 'blacklist_weight_decay_modules')
        had_whitelist = hasattr(self, 'whitelist_weight_decay_modules')

        if (not had_blacklist or not had_whitelist) and optim_args.get('weight_decay', 0.0) > 0.0:
            print(optim_args.get('weight_decay', 0.0), '!!')
            warnings.warn(f'Model does not specify: blacklist_weight_decay_modules or whitelist_weight_decay_modules, but weight_decay > 0.0. Weight decay will be applied to all parameters!')
            return self.parameters()
        elif optim_args.get('weight_decay', 0.0) > 0.0: # useing method from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py for selecting no_decay and decay params
            decay = set()
            no_decay = set()

            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    if pn.endswith('bias'):
                        no_decay.add(fpn)
                    elif isinstance(m, self.blacklist_weight_decay_modules):
                        no_decay.add(fpn)
                    elif isinstance(m, self.whitelist_weight_decay_modules):
                        decay.add(fpn)

            if hasattr(self, 'blacklist_param_names'): # add specific param names to blacklist
                for pn, p in self.named_parameters():
                    for bpn in self.blacklist_param_names:
                        if pn.endswith(bpn):
                            no_decay.add(pn)
                            break       


            param_dict = {pn: p for pn, p in self.named_parameters()}
            
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            optimizer_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optim_args['weight_decay']},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optimizer_groups
        else:
            return self.parameters()