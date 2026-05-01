import math

import torch
import torch.nn as nn
from torch import Tensor

from threedscriptors.model.preprocessing.geometric_preprocessor import RadialFilter


class PairBiasedSelfAttention(nn.Module):
    """
    Multi-head self-attention with an *additive* pair-tensor bias.

    Inputs
    ------
    S : (B, N, d_model)     atom/sequence embeddings
    P : (B, N, N, d_pair)   pair latent from PairTensorInit
    mask : (B, N) Bool      True for *real* atoms, False for padding

    Output
    ------
    S_out : (B, N, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_pair: int = 64,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "`d_model` must be divisible by `n_heads`"

        # --- Q, K, V projections ------------------------------------------------
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # --- pair → per-head bias ----------------------------------------------
        self.ln_pair = nn.LayerNorm(d_pair)
        self.pair2bias = nn.Linear(d_pair, n_heads, bias=False)
        nn.init.zeros_(self.pair2bias.weight)  # ← keeps first forward identical
        #   to baseline encoder
        # --- misc. --------------------------------------------------------------
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, S: Tensor, P: Tensor, mask: Tensor | None = None) -> Tensor:
        B, N, _ = S.shape

        # 1) project & reshape to (B, H, N, d_k)
        def _proj(W, x):
            return W(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
            #              (B, N, d) → (B, N, H, d_k) → (B, H, N, d_k)

        Q = _proj(self.W_q, S)
        K = _proj(self.W_k, S)
        V = _proj(self.W_v, S)  # (B, H, N, d_k) each

        # 2) scaled dot-product attention logits
        logits = torch.matmul(Q, K.transpose(-1, -2))  # (B, H, N, N)
        logits /= math.sqrt(self.d_k)

        # 3) add pair bias
        #    P: (B,N,N,d_p) → ln → (B,N,N,d_p) → linear → (B,N,N,H) → permute
        bias = self.pair2bias(self.ln_pair(P)).permute(0, 3, 1, 2)
        logits = logits + bias  # (B, H, N, N)

        # 4) padding mask  (True = pad, False = real atom)
        if mask is not None:
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(mask[:, None, None, :], neg_inf)  # mask keys
            logits = logits.masked_fill(mask[:, None, :, None], neg_inf)  # mask queries

        # 5) soft-max → weights
        attn_weights = torch.softmax(logits, dim=-1)  # (B, H, N, N)
        attn_weights = self.dropout(attn_weights)

        # 6) weighted sum
        S_head = torch.matmul(attn_weights, V)  # (B, H, N, d_k)
        S_head = S_head.transpose(1, 2)  # (B, N, H, d_k)
        S_head = S_head.reshape(B, N, self.d_model)  # (B, N, d_model)

        # 7) output projection
        S_out = self.W_o(S_head)  # (B, N, d_model)
        return S_out


class PairOuterProdUpdate(nn.Module):
    def __init__(self, d_model: int, d_pair: int, d_geo: int):
        super().__init__()
        # atom → pair part
        self.W_L = nn.Linear(d_model, d_pair, bias=False)
        self.W_R = nn.Linear(d_model, d_pair, bias=False)

        self.p_geo_filter = RadialFilter(d_geo, d_pair)

        self.layer_norm = nn.LayerNorm(d_pair)  # Unused

    def forward(self, S, P, p_geo, mask_pair):
        # The outer product of atomic descriptors thats gated by the rbf
        L = self.W_L(S)
        R = self.W_R(S)

        delta_outer_product = torch.sigmoid(self.p_geo_filter(p_geo)) * (
            L.unsqueeze(2) * R.unsqueeze(1)
        )  # (B,N,N,d_pair)

        P = P + 0.5 * (delta_outer_product + delta_outer_product.transpose(1, 2))

        P = P * mask_pair.unsqueeze(-1)

        return P
