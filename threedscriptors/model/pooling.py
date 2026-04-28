import math

import torch
import torch.nn.functional as F
from torch import nn


class MeanPool(nn.Module):
    def __init__(self, d_in: int, output_dropout: float | None = None):
        super().__init__()
        self.d_out = d_in
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout else None

    def forward(self, S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True for padded atoms, False for real atoms
        inverted_mask = ~mask
        inv_mask_expanded = inverted_mask.unsqueeze(-1).to(S.dtype)
        descriptor_sum = (S * inv_mask_expanded).sum(dim=1)
        counts = inv_mask_expanded.sum(dim=1)
        out = descriptor_sum / counts
        if self.output_dropout is not None:
            out = self.output_dropout(out)
        return out


class AttnPool(nn.Module):
    def __init__(
        self,
        d_in,
        d_out: int | None = None,
        n_heads=4,
        d_hidden=None,
        dropout=0.0,
        output_dropout: float | None = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_hidden = d_hidden or d_in
        assert self.d_hidden % n_heads == 0, "d_hidden must divide n_heads"

        self.d_k = self.d_hidden // n_heads
        self.d_out = d_out if d_out is not None else d_in
        self.scale = self.d_k**-0.5

        self.key_proj = nn.Linear(d_in, self.d_hidden, bias=False)
        self.query = nn.Parameter(torch.randn(n_heads, self.d_k))
        nn.init.xavier_uniform_(self.query)

        self.out_proj: nn.Module = (
            nn.Linear(d_in, self.d_out, bias=False)
            if self.d_out != d_in
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout else None

    def forward(self, x, pad_mask=None):
        """
        x        : (B, N, d_in)
        pad_mask : (B, N)  1 for padding, 0 for real tokens
        returns  : (B, d_out)
        """
        B, N, _ = x.shape

        k = self.key_proj(x)
        k = k.view(B, N, self.n_heads, self.d_k)

        logits = torch.einsum("bnhd,hd->bhn", k * self.scale, self.query)

        if pad_mask is not None:
            pad_exp = pad_mask[:, None, :].expand(B, self.n_heads, N)
            logits = logits.masked_fill(pad_exp == 1, float("-1e9"))

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        pooled = torch.einsum("bhn,bnd->bhd", attn, x)
        out = pooled.mean(dim=1)
        out = self.out_proj(out)
        if self.output_dropout is not None:
            out = self.output_dropout(out)
        return out


class PMAAggregator(nn.Module):
    """
    Pooling-by-(Multi)Head Attention with configurable value dimension.

    Inputs:
      S: (B, N, d_in)            set/sequence embeddings (e.g., atomic descriptors)
      padding_mask: (B, N) bool  True = PAD positions to ignore

    Returns:
      (B, d_out) single global vector (reduced over seeds)
    """

    def __init__(
        self,
        d_in: int,
        d_out,
        num_heads,
        head_dim,
        k_seeds,
        dropout,
        use_mlp,
        output_dropout: float | None = None,
    ):
        super().__init__()

        self.k = k_seeds
        self.H = num_heads
        self.use_mlp = use_mlp
        self.d_out = d_out

        d_qk_total = head_dim
        assert d_qk_total % num_heads == 0, "head_dim must be divisible by num_heads"
        self.d_k = d_qk_total // num_heads

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_v = d_out // num_heads

        self.num_seeds = k_seeds
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.ln_q = nn.LayerNorm(d_qk_total)
        self.ln_kv = nn.LayerNorm(d_in)

        self.seeds = nn.Parameter(torch.empty(1, self.num_seeds, d_qk_total))
        nn.init.xavier_uniform_(self.seeds)

        self.W_Q = nn.Linear(d_qk_total, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_in, num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_in, num_heads * self.d_v, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(d_out)

        self.mlp = None
        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_out, 4 * d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_out, d_out),
            )

        self.d_qk_total = d_qk_total
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout else None

    def forward(
        self, S: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        S: (B, N, d_in)
        padding_mask: (B, N) True for PAD to ignore
        """
        B, N, _ = S.shape
        H, d_k, d_v = self.H, self.d_k, self.d_v

        Z = self.ln_kv(S)
        seeds = self.seeds.expand(B, self.num_seeds, -1)
        seeds = self.ln_q(seeds)

        Q = self.W_Q(seeds).view(B, self.num_seeds, H, d_k).transpose(1, 2)
        K = self.W_K(Z).view(B, N, H, d_k).transpose(1, 2)
        V = self.W_V(Z).view(B, N, H, d_v).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(
                mask, torch.finfo(attn_logits.dtype).min
            )

        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        Y = torch.matmul(attn, V)
        Y = Y.transpose(1, 2).contiguous().view(B, self.num_seeds, H * d_v)

        Y = self.out_ln(Y)
        if self.mlp is not None:
            Y = Y + self.mlp(Y)

        G = torch.mean(Y, dim=1)

        if self.output_dropout is not None:
            G = self.output_dropout(G)

        return G
