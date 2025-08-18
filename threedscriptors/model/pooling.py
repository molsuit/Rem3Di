import torch
import torch.nn.functional as F
from torch import nn


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True for padded atoms, False for real atoms
        inverted_mask = ~mask
        # expand mask to match feature dimension
        inv_mask_expanded = inverted_mask.unsqueeze(-1).to(S.dtype)
        descriptor_sum = (S * inv_mask_expanded).sum(dim=1)
        counts = inv_mask_expanded.sum(dim=1)
        return descriptor_sum / counts



class AttnPool(nn.Module):
    def __init__(self, d_in, n_heads=4, d_hidden=None, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_hidden = d_hidden or d_in
        assert self.d_hidden % n_heads == 0, "d_hidden must divide n_heads"

        self.d_k = self.d_hidden // n_heads
        self.scale = self.d_k ** -0.5

        self.key_proj = nn.Linear(d_in, self.d_hidden, bias=False)
        self.query    = nn.Parameter(torch.randn(n_heads, self.d_k))
        nn.init.xavier_uniform_(self.query)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        """
        x        : (B, N, d_in)
        pad_mask : (B, N)  1 for padding, 0 for real tokens
        returns  : (B, d_in)
        """
        B, N, _ = x.shape

        # project keys and reshape for heads
        k = self.key_proj(x)                           # (B, N, d_hidden)
        k = k.view(B, N, self.n_heads, self.d_k)       # (B, N, H, d_k)

        # raw attention logits: (B, H, N)
        logits = torch.einsum('bnhd,hd->bhn', k * self.scale, self.query)

        if pad_mask is not None:
            # expand to (B, H, N)
            pad_exp = pad_mask[:, None, :].expand(B, self.n_heads, N)
            # mask out padded positions
            logits = logits.masked_fill(pad_exp == 1, float('-1e9'))

        # attention weights
        attn = F.softmax(logits, dim=-1)               # (B, H, N)
        attn = self.dropout(attn)

        # weighted sum in original feature space
        pooled = torch.einsum('bhn,bnd->bhd', attn, x) # (B, H, d_in)
        return pooled.mean(dim=1)

class AttnPoolNew(nn.Module):
    def __init__(self, d_in, n_heads=4, d_hidden=None, d_out=None, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_hidden = d_hidden or d_in
        assert self.d_hidden % n_heads == 0
        self.d_k = self.d_hidden // n_heads
        self.scale = self.d_k ** -0.5

        self.key_proj   = nn.Linear(d_in, self.d_hidden, bias=False)
        self.value_proj = nn.Linear(d_in, self.d_hidden, bias=False)
        self.query      = nn.Parameter(torch.empty(n_heads, self.d_k))
        nn.init.xavier_uniform_(self.query)

        self.out = nn.Linear(self.d_hidden, d_out or d_in, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):  # x: (B,N,d_in)
        B, N, _ = x.shape
        K = self.key_proj(x).view(B, N, self.n_heads, self.d_k)   # (B,N,H,d_k)
        V = self.value_proj(x).view(B, N, self.n_heads, self.d_k) # (B,N,H,d_k)

        logits = torch.einsum('bnhd,hd->bhn', K * self.scale, self.query)  # (B,H,N)
        if pad_mask is not None:
            logits = logits.masked_fill(pad_mask[:, None, :] == 1, float('-inf'))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        pooled = torch.einsum('bhn,bnhd->bhd', attn, V).contiguous()  # (B,H,d_k)
        pooled = pooled.view(B, self.d_hidden)                         # concat heads
        return self.out(pooled)




