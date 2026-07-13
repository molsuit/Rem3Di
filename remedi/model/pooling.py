import math

import torch
import torch.nn.functional as F
from torch import nn


class MeanPool(nn.Module):
    seq_len: int = 1

    def __init__(
        self,
        d_in: int,
        d_out: int | None = None,
        output_dropout: float | None = None,
    ):
        super().__init__()
        self.d_out = d_out if d_out is not None else d_in
        # Mirror AttnPool: project the pooled vector to the requested
        # descriptor dim so the downstream decoder cross-attention (wired for
        # `output_dim`) matches. Identity when no projection is needed.
        self.out_proj: nn.Module = (
            nn.Linear(d_in, self.d_out, bias=False)
            if self.d_out != d_in
            else nn.Identity()
        )
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout else None

    def forward(self, S: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True for padded atoms, False for real atoms.
        # Returns (B, 1, d_out) — a length-1 descriptor sequence.
        inverted_mask = ~mask
        inv_mask_expanded = inverted_mask.unsqueeze(-1).to(S.dtype)
        descriptor_sum = (S * inv_mask_expanded).sum(dim=1)
        counts = inv_mask_expanded.sum(dim=1)
        out = descriptor_sum / counts
        out = self.out_proj(out)
        if self.output_dropout is not None:
            out = self.output_dropout(out)
        return out.unsqueeze(1)


class AttnPool(nn.Module):
    seq_len: int = 1

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
        returns  : (B, 1, d_out) — length-1 descriptor sequence
        """
        B, N, _ = x.shape

        k = self.key_proj(x)
        k = k.view(B, N, self.n_heads, self.d_k)

        logits = torch.einsum("bnhd,hd->bhn", k * self.scale, self.query)

        if pad_mask is not None:
            pad_exp = pad_mask[:, None, :].expand(B, self.n_heads, N)
            logits = logits.masked_fill(pad_exp == 1, torch.finfo(logits.dtype).min)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        pooled = torch.einsum("bhn,bnd->bhd", attn, x)
        out = pooled.mean(dim=1)
        out = self.out_proj(out)
        if self.output_dropout is not None:
            out = self.output_dropout(out)
        return out.unsqueeze(1)


class PMAAggregator(nn.Module):
    """
    Pooling by Multihead Attention (Set Transformer, Lee et al. 2019).

    A bank of `k_seeds` learnable query vectors cross-attends to the (padded)
    set of input tokens. The result is a residual MAB block, optionally
    followed by a row-wise feed-forward residual. The k seed outputs are
    returned as a sequence so that downstream cross-attention can attend to
    each seed independently.

    Inputs
    ------
    S            : (B, N, d_in)
    padding_mask : (B, N) bool, True at PAD positions to ignore.

    Returns
    -------
    out : (B, k_seeds, d_out)
        A sequence of `k_seeds` summary tokens, each of dim `d_out`. Reduce
        downstream however suits the consumer (cross-attention over seeds /
        flatten / mean / etc.).

    Notes
    -----
    `head_dim` is the *per-head* Q/K dim (PyTorch convention). The total Q/K
    dim is `num_heads * head_dim`. The per-head value dim is fixed to
    `d_out / num_heads` so that, after head concatenation, the attention
    output matches the seed (residual stream) dim.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        head_dim: int,
        k_seeds: int,
        dropout: float = 0.0,
        use_mlp: bool = False,
        output_dropout: float | None = None,
    ):
        super().__init__()

        assert d_out % num_heads == 0, (
            f"d_out must be divisible by num_heads "
            f"(got d_out={d_out}, num_heads={num_heads})"
        )
        assert head_dim > 0 and num_heads > 0 and k_seeds > 0

        self.d_in = d_in
        self.d_out = d_out
        self.k_seeds = k_seeds
        self.seq_len = k_seeds
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_v = d_out // num_heads
        self.use_mlp = use_mlp
        self.scale = 1.0 / math.sqrt(head_dim)

        self.seeds = nn.Parameter(torch.empty(1, k_seeds, d_out))
        nn.init.xavier_uniform_(self.seeds)

        # Pre-norm transformer style: normalise inputs to each block.
        self.ln_seeds = nn.LayerNorm(d_out)
        self.ln_kv = nn.LayerNorm(d_in)

        self.W_Q = nn.Linear(d_out, num_heads * head_dim, bias=False)
        self.W_K = nn.Linear(d_in, num_heads * head_dim, bias=False)
        self.W_V = nn.Linear(d_in, num_heads * self.d_v, bias=False)
        self.W_O = nn.Linear(num_heads * self.d_v, d_out, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        if use_mlp:
            self.ln_ff = nn.LayerNorm(d_out)
            self.ff = nn.Sequential(
                nn.Linear(d_out, 4 * d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_out, d_out),
            )
        else:
            self.ln_ff = None
            self.ff = None

        self.output_dropout = nn.Dropout(output_dropout) if output_dropout else None

    def forward(
        self, S: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, _ = S.shape
        H, d_h, d_v = self.num_heads, self.head_dim, self.d_v

        seeds = self.seeds.expand(B, -1, -1)
        seeds_n = self.ln_seeds(seeds)
        Z = self.ln_kv(S)

        Q = self.W_Q(seeds_n).view(B, self.k_seeds, H, d_h).transpose(1, 2)
        K = self.W_K(Z).view(B, N, H, d_h).transpose(1, 2)
        V = self.W_V(Z).view(B, N, H, d_v).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            mask = padding_mask[:, None, None, :]
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        attn = logits.softmax(dim=-1)

        # When every key is masked, softmax-of-all-(-inf) collapses to a uniform
        # distribution (max-subtraction yields zeros), which would leak gradients
        # back through padded V projections. Zero those rows out explicitly.
        if padding_mask is not None:
            all_pad = padding_mask.all(dim=-1)
            if all_pad.any():
                attn = attn.masked_fill(all_pad[:, None, None, None], 0.0)

        attn = self.attn_drop(attn)

        head_out = torch.matmul(attn, V)
        head_out = head_out.transpose(1, 2).contiguous().view(B, self.k_seeds, H * d_v)

        # MAB residual + optional pre-norm rFF residual.
        Y = seeds + self.W_O(head_out)
        if self.ff is not None:
            assert self.ln_ff is not None
            Y = Y + self.ff(self.ln_ff(Y))

        if self.output_dropout is not None:
            Y = self.output_dropout(Y)
        return Y
