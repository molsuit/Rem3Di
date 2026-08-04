"""Pre-rewrite PMA aggregator, kept so published checkpoints stay loadable.

Checkpoints trained before the :class:`~remedi.model.pooling.PMAAggregator`
rewrite cannot be loaded by it, and cannot be converted into it either. Two
things changed that are not renames:

* ``head_dim`` meant the **total** Q/K width; it now means the **per-head**
  width, so the current module builds Q/K projections ``num_heads`` times
  wider. A ``(320, 320)`` ``W_Q`` cannot be reshaped into a ``(2560, 640)``
  one — the parameters do not exist.
* There is now an output projection ``W_O`` which the old module never had.

The old module also reduced over seeds internally, returning ``(B, d_out)``,
where the current one returns the full ``(B, k_seeds, d_out)`` sequence.

So this is not a compatibility shim over the new module — it is the old module,
preserved verbatim, selected by ``aggregator_type: pma_attention_legacy``. A
checkpoint loaded through it computes exactly the function it computed at
training time, which is what makes previously published numbers reproducible.

New work should use :class:`~remedi.model.pooling.PMAAggregator`.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PMAAggregatorLegacy(nn.Module):
    """Pooling by Multihead Attention, pre-rewrite semantics.

    Parameters
    ----------
    d_in
        Width of the input tokens.
    d_out
        Width of the pooled output. Must be divisible by ``num_heads``.
    num_heads
        Attention heads.
    head_dim
        **Total** Q/K width across all heads — not per-head. Must be divisible
        by ``num_heads``. This is the parameter whose meaning changed.
    k_seeds
        Number of learnable seed queries.
    dropout
        Dropout on the attention weights.
    use_mlp
        Add a residual feed-forward block after attention.

    Inputs
    ------
    S            : (B, N, d_in)
    padding_mask : (B, N) bool, True at PAD positions to ignore.

    Returns
    -------
    (B, d_out) — reduced over seeds by mean, unlike the current module which
    returns the seed sequence.
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
    ) -> None:
        super().__init__()

        d_qk_total = head_dim
        if d_qk_total % num_heads != 0:
            raise ValueError(
                f"head_dim is the TOTAL Q/K width here and must be divisible by "
                f"num_heads (got head_dim={head_dim}, num_heads={num_heads})"
            )
        if d_out % num_heads != 0:
            raise ValueError(
                f"d_out must be divisible by num_heads "
                f"(got d_out={d_out}, num_heads={num_heads})"
            )

        self.k = k_seeds
        self.num_seeds = k_seeds
        self.H = num_heads
        self.use_mlp = use_mlp
        self.d_k = d_qk_total // num_heads
        self.d_v = d_out // num_heads
        self.d_qk_total = d_qk_total
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.ln_q = nn.LayerNorm(d_qk_total)
        self.ln_kv = nn.LayerNorm(d_in)

        self.seeds = nn.Parameter(torch.empty(1, k_seeds, d_qk_total))
        nn.init.xavier_uniform_(self.seeds)

        self.W_Q = nn.Linear(d_qk_total, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_in, num_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_in, num_heads * self.d_v, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_ln = nn.LayerNorm(d_out)

        self.mlp = None
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(d_out, 4 * d_out),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_out, d_out),
            )

    def forward(
        self, S: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, _ = S.shape
        H, d_k, d_v = self.H, self.d_k, self.d_v

        Z = self.ln_kv(S)
        seeds = self.ln_q(self.seeds.expand(B, self.num_seeds, -1))

        Q = self.W_Q(seeds).view(B, self.num_seeds, H, d_k).transpose(1, 2)
        K = self.W_K(Z).view(B, N, H, d_k).transpose(1, 2)
        V = self.W_V(Z).view(B, N, H, d_v).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(
                mask, torch.finfo(attn_logits.dtype).min
            )

        attn = self.attn_drop(attn_logits.softmax(dim=-1))
        Y = torch.matmul(attn, V)
        Y = Y.transpose(1, 2).contiguous().view(B, self.num_seeds, H * d_v)

        Y = self.out_ln(Y)
        if self.mlp is not None:
            Y = Y + self.mlp(Y)

        return torch.mean(Y, dim=1)
