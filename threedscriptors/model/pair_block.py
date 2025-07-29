import torch
import torch.nn.functional as F
from torch import Tensor, nn

from threedscriptors.configuration.architecture_config import EncoderConfig
from threedscriptors.model.pair_biased_attention import (
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


class PairFFN(nn.Module):
    def __init__(self, d_pair: int, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(d_pair, d_pair)
        self.gate = nn.Linear(d_pair, d_pair, bias=False)
        nn.init.zeros_(self.gate.weight)  # zero-init → no effect at start
        self.dropout = nn.Dropout(dropout)

    def forward(self, P: Tensor) -> Tensor:
        delta = self.lin(F.silu(P))
        return P + self.dropout(torch.sigmoid(self.gate(P)) * delta)


class PairBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        d_pair: int,
        dim_feedforward,
        d_geo,
        dropout=0.1,
        pair_ffn: bool = True,
    ):
        super().__init__()

        # 1 attention sub-layer
        self.ln_s1 = nn.LayerNorm(embedding_dim)
        self.attn = PairBiasedSelfAttention(embedding_dim, num_heads, d_pair, dropout)

        # FFN sub-layer for atoms
        self.ln_s2 = nn.LayerNorm(embedding_dim)
        self.ffn_s = FeedForward(embedding_dim, dim_feedforward, dropout)

        # pair update as before
        self.pair_up = PairOuterProdUpdate(embedding_dim, d_pair, d_geo)

        # optional mini-FFN over the pair tensor
        self.pair_ffn = PairFFN(d_pair) if pair_ffn else nn.Identity()

    def forward(self, S, mask, P, p_geo, mask_pair):

        # Atom Representation Attention Update
        
        S = S + self.attn(self.ln_s1(S), P, mask)

        S = S + self.ffn_s(self.ln_s2(S))  # FFN Update

        # Pair Representation update

        P = self.pair_up(S, P, p_geo, mask_pair)  # Outer product update
        P = self.pair_ffn(P)  # Pair FFNN

        return S, P


class PairCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        d_pair: int,
        dim_feedforward,
        d_geo,
        dropout=0.1,
        pair_ffn: bool = True,
    ):
        super().__init__()

        # self attention sub-layer
        self.ln_s1 = nn.LayerNorm(embedding_dim)
        self.attn = PairBiasedSelfAttention(embedding_dim, num_heads, d_pair, dropout)

        # 2 Cross attention sublayer to the molecular_descriptor
        ...

        # 3 FFN sub-layer for atoms
        self.ln_s2 = nn.LayerNorm(embedding_dim)
        self.ffn_s = FeedForward(embedding_dim, dim_feedforward, dropout)

        # pair outer product update with the denoised embeddings
        self.pair_up = PairOuterProdUpdate(embedding_dim, d_pair, d_geo)

        # optional mini-FFN over the pair tensor
        self.pair_ffn = PairFFN(d_pair) if pair_ffn else nn.Identity()

    def forward(self, S, mask, P, p_geo, mask_pair):

        # Atom Representation Update

        S = S + self.attn(self.ln_s1(S), P, mask)

        # Attention based update
        S = S + self.ffn_s(self.ln_s2(S))  # FFN Update

        # Pair Representation update

        P = self.pair_up(S, P, p_geo, mask_pair)  # Outer product update
        P = self.pair_ffn(P)  # Pair FFNN

        return S, P
