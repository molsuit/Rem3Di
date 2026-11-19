import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from remedi.model.attention import key_padding_bias
from remedi.model.preprocessing.geometric_preprocessor import RadialFilter


class PairBiasedSelfAttention(nn.Module):
    """
    Multi-head self-attention with an *additive* pair-tensor bias.

    Inputs
    ------
    S : (B, N, d_model)     atom/sequence embeddings
    P : (B, N, N, d_pair)   pair latent from PairTensorInit
    mask : (B, N) Bool      True for padding, False for *real* atoms

    Output
    ------
    S_out : (B, N, d_model)

    Padded atoms are masked as keys; their own (query) rows are computed but
    meaningless, and every consumer masks them out.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_pair: int = 64,
        dropout_p: float = 0.1,
        pair_bias: bool = True,
    ):
        """``pair_bias=False``: plain self-attention, no pair input."""
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "`d_model` must be divisible by `n_heads`"

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # pair → per-head bias, zero-initialised so the first forward pass is
        # the plain attention
        self.ln_pair: nn.LayerNorm | None = None
        self.pair2bias: nn.Linear | None = None
        if pair_bias:
            self.ln_pair = nn.LayerNorm(d_pair)
            self.pair2bias = nn.Linear(d_pair, n_heads, bias=False)
            nn.init.zeros_(self.pair2bias.weight)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, S: Tensor, P: Tensor | None, mask: Tensor | None = None
    ) -> Tensor:
        B, N, _ = S.shape

        def heads(W: nn.Linear) -> Tensor:  # (B, N, d) -> (B, H, N, d_k)
            return W(S).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = heads(self.W_q), heads(self.W_k), heads(self.W_v)
        bias: Tensor | None = None
        if self.pair2bias is not None and self.ln_pair is not None:
            if P is None:
                raise ValueError("pair-biased attention needs the pair tensor")
            # (B, N, N, d_pair) -> (B, H, N, N)
            bias = self.pair2bias(self.ln_pair(P)).permute(0, 3, 1, 2).to(Q.dtype)
        if mask is not None:
            padding = key_padding_bias(mask, Q.dtype)
            bias = padding if bias is None else bias + padding
        attended = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.W_o(attended.transpose(1, 2).reshape(B, N, self.d_model))


class PairOuterProdUpdate(nn.Module):
    """Residual pair update from the atoms: a symmetrised, distance-gated
    product of two projections of the (layer-normalised) atom embeddings.

    The atoms are normalised first (as in AlphaFold's outer product mean), so
    the update, which is quadratic in them, does not grow with the norm of the
    residual stream.
    """

    def __init__(self, d_model: int, d_pair: int, d_geo: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # atom → pair part
        self.W_L = nn.Linear(d_model, d_pair, bias=False)
        self.W_R = nn.Linear(d_model, d_pair, bias=False)

        self.p_geo_filter = RadialFilter(d_geo, d_pair)

    def forward(self, S, P, p_geo, mask_pair):
        S = self.norm(S)
        L = self.W_L(S)
        R = self.W_R(S)

        delta_outer_product = torch.sigmoid(self.p_geo_filter(p_geo)) * (
            L.unsqueeze(2) * R.unsqueeze(1)
        )  # (B,N,N,d_pair)

        P = P + 0.5 * (delta_outer_product + delta_outer_product.transpose(1, 2))

        P = P * mask_pair.unsqueeze(-1)

        return P
