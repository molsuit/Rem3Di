import math

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
        self.scale = self.d_k**-0.5

        self.key_proj = nn.Linear(d_in, self.d_hidden, bias=False)
        self.query = nn.Parameter(torch.randn(n_heads, self.d_k))
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
        k = self.key_proj(x)  # (B, N, d_hidden)
        k = k.view(B, N, self.n_heads, self.d_k)  # (B, N, H, d_k)

        # raw attention logits: (B, H, N)
        logits = torch.einsum("bnhd,hd->bhn", k * self.scale, self.query)

        if pad_mask is not None:
            # expand to (B, H, N)
            pad_exp = pad_mask[:, None, :].expand(B, self.n_heads, N)
            # mask out padded positions
            logits = logits.masked_fill(pad_exp == 1, float("-1e9"))

        # attention weights
        attn = F.softmax(logits, dim=-1)  # (B, H, N)
        attn = self.dropout(attn)

        # weighted sum in original feature space
        pooled = torch.einsum("bhn,bnd->bhd", attn, x)  # (B, H, d_in)
        return pooled.mean(dim=1)


class PMAAggregator(nn.Module):
    """
    Pooling-by-(Multi)Head Attention with configurable value dimension.

    Inputs:
      S: (B, N, d_in)            set/sequence embeddings (e.g., atomic descriptors)
      padding_mask: (B, N) bool  True = PAD positions to ignore

    Returns:
      (B, out_dim) single global vector (reduced over seeds)
    """

    def __init__(self, d_in: int, d_out, num_heads, head_dim, k_seeds, dropout, use_mlp):
        super().__init__()


        self.k = k_seeds
        self.H = num_heads
        self.use_mlp = use_mlp

        d_qk_total = head_dim
        assert d_qk_total % num_heads == 0, "head_dim must be divisible by num_heads"
        self.d_k = d_qk_total // num_heads  # per-head key/query size


        assert d_out % num_heads == 0, "d_v_out must be divisible by num_heads"
        self.d_v = d_out // num_heads  # per-head value size

        self.num_seeds = k_seeds
        self.scale = 1.0 / math.sqrt(self.d_k)

        # LayerNorm on inputs
        self.ln_q = nn.LayerNorm(d_qk_total)  # seeds live in d_qk_total
        self.ln_kv = nn.LayerNorm(d_in)  # keys/values come from d_in

        # Learnable seeds (queries) in Q/K space
        self.seeds = nn.Parameter(torch.empty(1, self.num_seeds, d_qk_total))
        nn.init.xavier_uniform_(self.seeds)

        # Linear projections
        self.W_Q = nn.Linear(d_qk_total, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_in, num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_in, num_heads * self.d_v, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(d_out)

        # Optional tiny FFN (+residual) before reduction
        self.mlp = None
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_out, 4 * d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_out, d_out),
            )

        self.d_qk_total = d_qk_total

    def forward(
        self, S: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        S: (B, N, d_in)
        padding_mask: (B, N) True for PAD to ignore
        """
        B, N, _ = S.shape
        H, d_k, d_v = self.H, self.d_k, self.d_v

        # Norm + seeds
        Z = self.ln_kv(S)  # (B, N, d_in)
        seeds = self.seeds.expand(B, self.num_seeds, -1)  # (B, k, d_qk_total)
        seeds = self.ln_q(seeds)

        # Projections
        Q = (
            self.W_Q(seeds).view(B, self.num_seeds, H, d_k).transpose(1, 2)
        )  # (B, H, k, d_k)
        K = self.W_K(Z).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        V = self.W_V(Z).view(B, N, H, d_v).transpose(1, 2)  # (B, H, N, d_v)

        # Attention logits: (B, H, k, N)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            # padding_mask: True where we should ignore
            mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_logits = attn_logits.masked_fill(
                mask, torch.finfo(attn_logits.dtype).min
            )

        # Weights + dropout
        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values: (B, H, k, d_v)
        Y = torch.matmul(attn, V)

        # Merge heads → (B, k, H*d_v = d_v_out)
        Y = Y.transpose(1, 2).contiguous().view(B, self.num_seeds, H * d_v)

        # Post-attention LN (+ optional MLP residual)
        Y = self.out_ln(Y)
        if self.mlp is not None:
            Y = Y + self.mlp(Y)

        G = torch.mean(Y, dim=1)

        return G  # (B, out_dim)
