"""The transformer block shared by the encoder and the decoder.

A simplified AlphaFold-style block: the pair representation biases the atom
self-attention, and (optionally) the atoms update the pair representation
through a gated outer product followed by a small transition MLP. There are no
triangle updates.
"""

import torch.nn.functional as F
from torch import Tensor, nn

from remedi.model.multihead_self_attention import MultiHeadCrossAttention
from remedi.model.pair_biased_attention import (
    PairBiasedSelfAttention,
    PairOuterProdUpdate,
)


class FeedForward(nn.Module):
    """GeGLU-style FFN (a bit better than ReLU/GELU + Linear)."""

    def __init__(self, d_model: int, d_hidden: int | None = None, dropout=0.1):
        super().__init__()
        d_hidden = d_hidden or 4 * d_model  # usual width factor
        self.proj_in = nn.Linear(d_model, d_hidden * 2)  # 2x for GEGLU
        self.proj_out = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x_in = self.proj_in(x)  # (…, 2 d_hidden)
        x_g, x_h = x_in.chunk(2, dim=-1)
        x = F.gelu(x_g) * x_h  # GEGLU
        return self.proj_out(self.dropout(x))


class PairTransition(nn.Module):
    """Pre-norm residual MLP on the pair representation.

    The last layer is zero-initialised, so the block is the identity at the
    start of training.
    """

    def __init__(self, d_pair: int, dropout: float = 0.1, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.proj_in = nn.Linear(d_pair, expansion * d_pair)
        self.proj_out = nn.Linear(expansion * d_pair, d_pair)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, P: Tensor) -> Tensor:
        delta = self.proj_out(F.silu(self.proj_in(self.norm(P))))
        return P + self.dropout(delta)


class PairBlock(nn.Module):
    """Pair-biased self-attention, optional cross-attention, feed-forward, and
    an optional pair update.

    ``descriptor_dimension`` set (the decoder): a cross-attention to the
    molecular descriptor tokens follows the self-attention.
    ``update_pair=False`` leaves the pair tensor as it is: the last layer (whose
    pair output nothing reads), or every layer of a model that uses the initial
    pair representation as a static attention bias.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        d_pair: int,
        dim_feedforward: int,
        d_geo: int,
        dropout: float = 0.1,
        descriptor_dimension: int | None = None,
        update_pair: bool = True,
    ):
        super().__init__()

        self.self_attention_norm = nn.LayerNorm(embedding_dim)
        self.self_attention = PairBiasedSelfAttention(
            embedding_dim, num_heads, d_pair, dropout
        )

        self.cross_attention_norm: nn.LayerNorm | None = None
        self.cross_attention: MultiHeadCrossAttention | None = None
        if descriptor_dimension is not None:
            self.cross_attention_norm = nn.LayerNorm(embedding_dim)
            self.cross_attention = MultiHeadCrossAttention(
                embedding_dim, descriptor_dimension, num_heads, dropout
            )

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, dim_feedforward, dropout)

        self.pair_update: PairOuterProdUpdate | None = None
        self.pair_transition: PairTransition | None = None
        if update_pair:
            self.pair_update = PairOuterProdUpdate(embedding_dim, d_pair, d_geo)
            self.pair_transition = PairTransition(d_pair, dropout)

    def forward(
        self,
        S: Tensor,
        padding_mask: Tensor,
        P: Tensor,
        p_geo: Tensor,
        mask_pair: Tensor,
        descriptor_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        S = S + self.self_attention(self.self_attention_norm(S), P, padding_mask)

        if self.cross_attention is not None and self.cross_attention_norm is not None:
            if descriptor_tokens is None:
                raise ValueError("a block with cross-attention needs descriptor_tokens")
            S = S + self.cross_attention(
                self.cross_attention_norm(S), descriptor_tokens
            )

        S = S + self.feed_forward(self.feed_forward_norm(S))

        # Both pair updates are residual already (they return P + delta).
        if self.pair_update is not None and self.pair_transition is not None:
            P = self.pair_update(S, P, p_geo, mask_pair)
            P = self.pair_transition(P)

        return S, P
