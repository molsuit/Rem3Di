"""Tests for LoRA adapter injection.

All CPU-only: no checkpoint, no zarr, no GPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from remedi.training.lora import (
    LoraLinear,
    count_trainable_parameters,
    freeze_non_lora_parameters,
    inject_lora_adapters,
)


class _Attn(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)


class _Block(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.attn = _Attn(d)
        self.ffn = nn.Linear(d, d)


class _Aggregator(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)


class _Encoder(nn.Module):
    """Mirrors the real name layout: layers.N.attn.W_*, aggregator.W_*."""

    def __init__(self, n_layers: int = 3, d: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Block(d) for _ in range(n_layers))
        self.aggregator = _Aggregator(d)
        self.head = nn.Linear(d, 1)


def test_adapter_is_identity_at_init():
    """B is zero-initialised, so a fresh adapter must reproduce the base exactly."""
    torch.manual_seed(0)
    base = nn.Linear(16, 32, bias=True)
    wrapped = LoraLinear(base, r=4, alpha=8.0, dropout=0.0).eval()
    x = torch.randn(5, 16)
    torch.testing.assert_close(wrapped(x), base(x), rtol=0, atol=0)


def test_injection_hits_every_target():
    n_layers = 3
    model = _Encoder(n_layers=n_layers)
    counts = inject_lora_adapters(model, r=4, return_per_target=True)

    for target in (".attn.W_q", ".attn.W_k", ".attn.W_v", ".attn.W_o"):
        assert counts[target] == n_layers, f"{target} -> {counts[target]}"
    # The aggregator projections are the ones that silently went missing when
    # this module was named `.pool.*`; assert each is hit exactly once.
    for target in (".aggregator.W_Q", ".aggregator.W_K", ".aggregator.W_V",
                   ".aggregator.W_O"):
        assert counts[target] == 1, f"{target} -> {counts[target]}"

    assert sum(counts.values()) == 4 * n_layers + 4
    assert isinstance(model.aggregator.W_Q, LoraLinear)
    assert not isinstance(model.ffn if hasattr(model, "ffn") else model.head, LoraLinear)


def test_a_target_that_matches_nothing_raises():
    """A partial match is the dangerous case: it must fail loudly, not quietly."""
    model = _Encoder()
    with pytest.raises(RuntimeError, match="matched no modules"):
        inject_lora_adapters(model, target_substrings=(".attn.W_q", ".pool.W_Q"))


def test_freeze_leaves_only_adapters_trainable():
    model = _Encoder()
    inject_lora_adapters(model, r=4)
    freeze_non_lora_parameters(model)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "nothing left trainable"
    assert all(n.endswith(".A.weight") or n.endswith(".B.weight") for n in trainable)


def test_also_train_keeps_the_head_trainable():
    """Without also_train the head is frozen and training silently does nothing."""
    model = _Encoder()
    inject_lora_adapters(model, r=4)
    freeze_non_lora_parameters(model, also_train=("head",))

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert "head.weight" in trainable
    assert "head.bias" in trainable


def test_trainable_fraction_is_small():
    model = _Encoder(n_layers=4, d=64)
    inject_lora_adapters(model, r=4)
    freeze_non_lora_parameters(model)
    trainable, total = count_trainable_parameters(model)
    assert 0 < trainable / total < 0.15, f"{trainable}/{total}"


def test_scaling_decouples_rank_from_learning_rate():
    base = nn.Linear(8, 8, bias=False)
    assert LoraLinear(base, r=4, alpha=32.0).scaling == 8.0
    assert LoraLinear(base, r=16, alpha=32.0).scaling == 2.0


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="rank must be positive"):
        LoraLinear(nn.Linear(4, 4), r=0)
