import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from remedi.model.attention import key_padding_bias


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention.

    Inputs
    ------
    S : (B, N, d_model)     atom/sequence embeddings
    mask : (B, N) Bool      True for padding, False for *real* atoms

    Output
    ------
    S_out : (B, N, d_model)

    Padded atoms are masked as keys; their own (query) rows are computed but
    meaningless.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "`d_model` must be divisible by `n_heads`"

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, S: Tensor, mask: Tensor | None = None) -> Tensor:
        B, N, _ = S.shape

        def heads(W: nn.Linear) -> Tensor:  # (B, N, d) -> (B, H, N, d_k)
            return W(S).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = heads(self.W_q), heads(self.W_k), heads(self.W_v)
        attended = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None if mask is None else key_padding_bias(mask, Q.dtype),
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.W_o(attended.transpose(1, 2).reshape(B, N, self.d_model))


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention from atoms to a (set of) molecular descriptors.

    Inputs
    ------
    S : (B, N, d_model)            atom/sequence embeddings
    M : (B, L, d_descriptor)       descriptor sequence (e.g. PMA seeds);
                                   single-vector aggregators emit L=1.

    Output
    ------
    S_out : (B, N, d_model)

    Every descriptor token is a valid key, so nothing is masked; the rows of
    padded atoms are computed but meaningless, and every consumer masks them.
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

    def _heads(self, W: nn.Linear, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        return W(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, S: Tensor, M: Tensor) -> Tensor:
        attended = F.scaled_dot_product_attention(
            self._heads(self.W_q, S),  # (B, H, N, d_k)
            self._heads(self.W_k, M),  # (B, H, L, d_k)
            self._heads(self.W_v, M),  # (B, H, L, d_k)
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.W_o(attended.transpose(1, 2).reshape(S.size(0), S.size(1), -1))
