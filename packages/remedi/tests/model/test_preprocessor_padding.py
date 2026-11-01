"""Lightweight tests for the pad_multiple plumbing on PreprocessorWithAtomicEmbedding.

The full forward path needs a real MACE model; here we just verify that the
`pad_multiple` attribute is wired into the constructor, validated, and used by
the padding-length helper. The math itself is covered by
test_samplers::test_quantize_pad_length_*.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from remedi.model.preprocessing.preprocessing import (
    PreprocessorWithAtomicEmbedding,
)
from remedi.training.data.samplers import quantize_pad_length


class _StubMaceModel:
    r_max = 5.0


def _make_preprocessor(**kwargs) -> PreprocessorWithAtomicEmbedding:
    return PreprocessorWithAtomicEmbedding(
        mace_model=_StubMaceModel(),
        atomic_preprocessor=nn.Identity(),
        geometric_preprocessor=nn.Identity(),
        **kwargs,
    )


def test_default_pad_multiple_is_one():
    p = _make_preprocessor()
    assert p.pad_multiple == 1


def test_pad_multiple_is_stored():
    p = _make_preprocessor(pad_multiple=8)
    assert p.pad_multiple == 8


def test_pad_multiple_zero_rejected():
    with pytest.raises(ValueError, match="positive"):
        _make_preprocessor(pad_multiple=0)


def test_pad_multiple_negative_rejected():
    with pytest.raises(ValueError, match="positive"):
        _make_preprocessor(pad_multiple=-4)


def test_pad_multiple_post_construction_assignment():
    """Training script sets pad_multiple after building the bundle."""
    p = _make_preprocessor()
    p.pad_multiple = 16
    assert p.pad_multiple == 16


def test_quantize_matches_expected_buckets():
    """Sanity-check the buckets users will configure with bucketed batching."""
    # Atom counts in the user's TMQM histogram peak around 50-60 with tail to 200.
    # With pad_multiple=8, distinct compiled shapes are bounded.
    quantized = sorted({quantize_pad_length(n, 8) for n in range(1, 201)})
    assert max(quantized) == 200  # 200 is already a multiple of 8
    assert len(quantized) == 200 // 8  # 25 distinct shapes vs ~200 raw lengths
