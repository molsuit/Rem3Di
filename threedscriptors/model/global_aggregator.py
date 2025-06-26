import torch
import torch.nn.functional as F
from torch import nn

from threedscriptors.configuration.architecture_config import GlobalAggregatorConfig


class AttnPool(nn.Module):
    def __init__(self, d_in, n_heads=4, d_hidden=None, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_hidden = d_hidden or d_in        # allow same size
        assert self.d_hidden % n_heads == 0, "d_hidden must divide n_heads"

        self.d_k = self.d_hidden // n_heads
        self.scale = self.d_k ** -0.5

        self.key_proj = nn.Linear(d_in, self.d_hidden, bias=False)

        # learnable query per head
        self.query = nn.Parameter(torch.randn(n_heads, self.d_k))
        nn.init.xavier_uniform_(self.query)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
            """
            x : (B, N, d_in)
            returns : (B, d_in)
            """
            B, N, _ = x.shape

            # ---- keys ----------------------------------------------------------
            k = self.key_proj(x)                           # (B, N, d_hidden)
            k = k.view(B, N, self.n_heads, self.d_k)       # (B, N, H, d_k)

            # ---- attention logits ---------------------------------------------
            # einsum expects axes  b n h d  vs  h d
            logits = torch.einsum('bnhd,hd->bhn', k * self.scale, self.query)

            attn = F.softmax(logits, dim=-1)                  # (B, H, N)
            attn = self.dropout(attn)

            # ---- weighted sum in original feature space -----------------------
            pooled = torch.einsum('bhn,bnd->bhd', attn, x)    # (B, H, d_in)
            return pooled.mean(dim=1)



class GlobalAggregator(nn.Module):
    def __init__(self, global_aggregator_config: GlobalAggregatorConfig):
        super().__init__()
        self.config = global_aggregator_config

        self.pool = AttnPool(self.config.input_dim, d_hidden=self.config.input_dim, n_heads=4, dropout= 0.2)



        self.config.output_dim = (
            self.config.input_dim
        )  # len(self.aggregation_fns) * self.config.input_dim



    def forward(self, x):
        out = self.pool(x)

        #out = torch.mean(x, dim = 1)
        # intermediates = [f(x, dim=1) for f in self.aggregation_fns]
        # out = cat(intermediates, dim=-1)

        #out = self.dropout(out)
        return out
