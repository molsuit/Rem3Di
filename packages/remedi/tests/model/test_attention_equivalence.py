"""The attentions run on ``scaled_dot_product_attention`` and match the
hand-written attention they replaced, in eval mode, on every real atom.

The reference functions below are the replaced implementations' forward passes,
kept as test oracles. Rows of padded *query* atoms may differ (the old code
spread them uniformly over every key, the new one over the real keys); they are
discarded downstream (masked pooling, masked losses, masked pair updates), so
only real rows are compared.
"""

from __future__ import annotations

import math

import pytest
import torch

from remedi.model.multihead_self_attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from remedi.model.pair_biased_attention import PairBiasedSelfAttention
from remedi.model.pooling import AttnPool, PMAAggregator

BATCH, ATOMS, D_MODEL, HEADS, D_PAIR = 3, 7, 32, 4, 8


def _padding() -> torch.Tensor:
    """True marks padding: 7, 4 and 1 real atoms."""
    counts = torch.tensor([7, 4, 1])
    return torch.arange(ATOMS)[None, :] >= counts[:, None]


def _heads(linear: torch.nn.Linear, x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, _ = x.shape
    return linear(x).view(batch, length, heads, -1).transpose(1, 2)


def _reference_pair_biased(module: PairBiasedSelfAttention, S, P, mask) -> torch.Tensor:
    Q, K, V = (_heads(W, S, HEADS) for W in (module.W_q, module.W_k, module.W_v))
    logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(module.d_k)
    logits = logits + module.pair2bias(module.ln_pair(P)).permute(0, 3, 1, 2)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(mask[:, None, None, :], neg_inf)
    logits = logits.masked_fill(mask[:, None, :, None], neg_inf)
    heads = torch.matmul(torch.softmax(logits, dim=-1), V)
    return module.W_o(heads.transpose(1, 2).reshape(S.shape))


def _reference_cross(module: MultiHeadCrossAttention, S, M, mask) -> torch.Tensor:
    Q = _heads(module.W_q, S, HEADS)
    K = _heads(module.W_k, M, HEADS)
    V = _heads(module.W_v, M, HEADS)
    logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(module.d_k)
    logits = logits.masked_fill(mask[:, None, :, None], torch.finfo(logits.dtype).min)
    heads = torch.matmul(torch.softmax(logits, dim=-1), V)
    return module.W_o(heads.transpose(1, 2).reshape(S.shape[0], S.shape[1], -1))


def _reference_attention_pool(module: AttnPool, x, pad_mask) -> torch.Tensor:
    batch, atoms, _ = x.shape
    keys = module.key_proj(x).view(batch, atoms, module.n_heads, module.d_k)
    logits = torch.einsum("bnhd,hd->bhn", keys * module.d_k**-0.5, module.query)
    logits = logits.masked_fill(
        pad_mask[:, None, :].expand(batch, module.n_heads, atoms),
        torch.finfo(logits.dtype).min,
    )
    pooled = torch.einsum("bhn,bnd->bhd", torch.softmax(logits, dim=-1), x)
    return module.out_proj(pooled.mean(dim=1)).unsqueeze(1)


def _reference_pma(module: PMAAggregator, S, padding_mask) -> torch.Tensor:
    batch = S.shape[0]
    seeds = module.seeds.expand(batch, -1, -1)
    Z = module.ln_kv(S)
    Q = _heads(module.W_Q, module.ln_seeds(seeds), module.num_heads)
    K = _heads(module.W_K, Z, module.num_heads)
    V = _heads(module.W_V, Z, module.num_heads)
    logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(module.head_dim)
    logits = logits.masked_fill(
        padding_mask[:, None, None, :], torch.finfo(logits.dtype).min
    )
    attention = logits.softmax(dim=-1)
    all_padded = padding_mask.all(dim=-1)
    attention = attention.masked_fill(all_padded[:, None, None, None], 0.0)
    heads = (
        torch.matmul(attention, V).transpose(1, 2).reshape(batch, module.k_seeds, -1)
    )
    output = seeds + module.W_O(heads)
    if module.ff is not None:
        assert module.ln_ff is not None
        output = output + module.ff(module.ln_ff(output))
    return output


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_pair_biased_attention_matches_the_reference(dtype: torch.dtype) -> None:
    module = PairBiasedSelfAttention(D_MODEL, HEADS, D_PAIR, dropout_p=0.3)
    torch.nn.init.normal_(module.pair2bias.weight)  # zero-initialised otherwise
    module = module.to(dtype).eval()
    S = torch.randn(BATCH, ATOMS, D_MODEL, dtype=dtype)
    P = torch.randn(BATCH, ATOMS, ATOMS, D_PAIR, dtype=dtype)
    mask = _padding()
    with torch.no_grad():
        new = module(S, P, mask)
        old = _reference_pair_biased(module, S, P, mask)
    real = ~mask
    torch.testing.assert_close(new[real], old[real])
    assert torch.isfinite(new).all()


def test_cross_attention_matches_the_reference() -> None:
    module = MultiHeadCrossAttention(D_MODEL, 24, HEADS, dropout=0.3).eval()
    S = torch.randn(BATCH, ATOMS, D_MODEL)
    M = torch.randn(BATCH, 5, 24)
    mask = _padding()
    with torch.no_grad():
        new = module(S, M)
        old = _reference_cross(module, S, M, mask)
    real = ~mask
    torch.testing.assert_close(new[real], old[real])


def test_attention_pool_matches_the_reference() -> None:
    module = AttnPool(D_MODEL, d_out=16, n_heads=HEADS, dropout=0.3).eval()
    x = torch.randn(BATCH, ATOMS, D_MODEL)
    mask = _padding()
    with torch.no_grad():
        torch.testing.assert_close(
            module(x, mask), _reference_attention_pool(module, x, mask)
        )


@pytest.mark.parametrize("use_mlp", [False, True])
def test_pma_matches_the_reference_also_with_a_fully_padded_row(use_mlp: bool) -> None:
    module = PMAAggregator(
        D_MODEL,
        16,
        num_heads=HEADS,
        head_dim=8,
        k_seeds=3,
        dropout=0.3,
        use_mlp=use_mlp,
    ).eval()
    S = torch.randn(BATCH, ATOMS, D_MODEL)
    mask = _padding()
    mask[2] = True  # a molecule without a single real atom
    with torch.no_grad():
        torch.testing.assert_close(module(S, mask), _reference_pma(module, S, mask))


def test_dropout_acts_only_in_training() -> None:
    module = PairBiasedSelfAttention(D_MODEL, HEADS, D_PAIR, dropout_p=0.5)
    S = torch.randn(BATCH, ATOMS, D_MODEL)
    P = torch.randn(BATCH, ATOMS, ATOMS, D_PAIR)
    mask = _padding()
    module.eval()
    with torch.no_grad():
        torch.testing.assert_close(module(S, P, mask), module(S, P, mask))
    module.train()
    with torch.no_grad():
        assert not torch.allclose(module(S, P, mask), module(S, P, mask))


def _reference_self_attention(module: MultiHeadSelfAttention, S, mask) -> torch.Tensor:
    Q, K, V = (_heads(W, S, HEADS) for W in (module.W_q, module.W_k, module.W_v))
    logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(module.d_k)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(mask[:, None, None, :], neg_inf)
    logits = logits.masked_fill(mask[:, None, :, None], neg_inf)
    heads = torch.matmul(torch.softmax(logits, dim=-1), V)
    return module.W_o(heads.transpose(1, 2).reshape(S.shape))


def test_self_attention_matches_the_reference() -> None:
    module = MultiHeadSelfAttention(D_MODEL, HEADS, dropout=0.3).eval()
    S = torch.randn(BATCH, ATOMS, D_MODEL)
    mask = _padding()
    with torch.no_grad():
        new = module(S, mask)
        old = _reference_self_attention(module, S, mask)
    real = ~mask
    torch.testing.assert_close(new[real], old[real])
