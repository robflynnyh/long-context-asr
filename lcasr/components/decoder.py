import apex, torch.nn as nn, torch.nn.functional as F
DEFAULT_NORM = apex.normalization.FusedRMSNorm
from einops import rearrange

class ASRLinearSCDecoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            vocab_size, 
            norm=False, 
            expansion=1,
            norm_fn=DEFAULT_NORM
        ):
        super().__init__()
        # Add 1 for blank char
        self.num_classes = vocab_size + 1
        self.expansion = expansion
        assert expansion > 0
        self.ff = nn.Linear(d_model, self.num_classes * expansion)
        self.reprojection = nn.Linear(self.num_classes * expansion, d_model)
        self.norm = norm_fn(d_model) if norm else nn.Identity()
 

    def forward(self, x, logits=False):
        x_norm = self.norm(x)
        x = self.ff(x_norm)
        if self.expansion > 1: x = rearrange(x, 'b t (d e) -> b (t e) d', e=self.expansion)
        x = F.log_softmax(x, dim=-1) if not logits else x
        return x     

    def project_back(self, x):
        if self.expansion > 1: x = rearrange(x, 'b (t e) d -> b t (d e)', e=self.expansion)
        return self.reprojection(x)

    def integrate_projections(self, x, proj1):
        return x + proj1