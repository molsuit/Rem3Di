import math

import torch
import torch.nn as nn
from torch import Tensor


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with an *additive* pair-tensor bias.

    Inputs
    ------
    S : (B, N, d_model)     atom/sequence embeddings
    mask : (B, N) Bool      True for *real* atoms, False for padding

    Output
    ------
    S_out : (B, N, d_model)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
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

        self.dropout = nn.Dropout(dropout)

    def forward(self, S: Tensor, mask: Tensor | None = None) -> Tensor:
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

        # 3) padding mask  (True = pad, False = real atom)
        if mask is not None:
            neg_inf = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(mask[:, None, None, :], neg_inf)  # mask keys
            logits = logits.masked_fill(mask[:, None, :, None], neg_inf)  # mask queries

        # 4) soft-max → weights
        attn_weights = torch.softmax(logits, dim=-1)  # (B, H, N, N)
        attn_weights = self.dropout(attn_weights)

        # 6) weighted sum
        S_head = torch.matmul(attn_weights, V)  # (B, H, N, d_k)
        S_head = S_head.transpose(1, 2)  # (B, N, H, d_k)
        S_head = S_head.reshape(B, N, self.d_model)  # (B, N, d_model)

        # 7) output projection
        S_out = self.W_o(S_head)  # (B, N, d_model)
        return S_out


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention from atoms to a (set of) molecular descriptors.

    Inputs
    ------
    S : (B, N, d_model)            atom/sequence embeddings
    M : (B, L, d_descriptor)       descriptor sequence (e.g. PMA seeds).
                                   A 2D `(B, d_descriptor)` is also accepted
                                   for single-vector aggregators and treated
                                   as L=1.
    mask : (B, N) Bool             True for padding (atom side)

    Output
    ------
    S_out : (B, N, d_model)
    """

    def __init__(self, d_model=256, d_descriptor=256, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_descriptor, d_model, bias=False)
        self.W_v = nn.Linear(d_descriptor, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _proj(self, W, x):
        B, L, _ = x.shape
        return W(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, S, M, mask=None):
        # M is always (B, L, d_descriptor); aggregators that emit a single
        # vector use L=1.
        Q = self._proj(self.W_q, S)  # (B, H, N, d_k)
        K = self._proj(self.W_k, M)  # (B, H, L, d_k)
        V = self._proj(self.W_v, M)  # (B, H, L, d_k)

        logits = torch.matmul(Q, K.transpose(-1, -2))  # (B, H, N, L)
        logits = logits / math.sqrt(self.d_k)

        if mask is not None:
            # mask is on the *query* (atom) side, not the descriptor side.
            logits = logits.masked_fill(
                mask[:, None, :, None], torch.finfo(logits.dtype).min
            )

        attn = self.dropout(torch.softmax(logits, dim=-1))

        S_head = torch.matmul(attn, V)  # (B, H, N, d_k)
        S_head = S_head.transpose(1, 2).contiguous().view(S.size(0), S.size(1), -1)

        return self.W_o(S_head)
