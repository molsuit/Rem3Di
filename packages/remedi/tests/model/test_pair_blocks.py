"""Pair-representation updates of the encoder and decoder blocks."""

from __future__ import annotations

import pytest
import torch

from remedi.model.decoder import DecoderPairBlock
from remedi.model.pair_block import PairBlock

EMBEDDING_DIMENSION = 8
PAIR_DIMENSION = 4
RADIAL_BASIS_DIMENSION = 6
NUMBER_OF_ATOMS = 5


def _block_inputs():
    generator = torch.Generator().manual_seed(0)
    atoms = torch.randn(1, NUMBER_OF_ATOMS, EMBEDDING_DIMENSION, generator=generator)
    pairs = torch.randn(
        1, NUMBER_OF_ATOMS, NUMBER_OF_ATOMS, PAIR_DIMENSION, generator=generator
    )
    pairs = 0.5 * (pairs + pairs.transpose(1, 2))
    radial_basis = torch.rand(
        1,
        NUMBER_OF_ATOMS,
        NUMBER_OF_ATOMS,
        RADIAL_BASIS_DIMENSION,
        generator=generator,
    )
    padding_mask = torch.zeros(1, NUMBER_OF_ATOMS, dtype=torch.bool)
    pair_mask = torch.ones(1, NUMBER_OF_ATOMS, NUMBER_OF_ATOMS, dtype=torch.bool)
    descriptor = torch.randn(1, 1, EMBEDDING_DIMENSION, generator=generator)
    return atoms, pairs, radial_basis, padding_mask, pair_mask, descriptor


def _silence_pair_deltas(block) -> None:
    """Zero the outer-product and pair-FFN weights so both pair updates add a
    zero delta: a correct residual block then leaves the pair tensor as is."""
    pair_update = block.pair_update if hasattr(block, "pair_update") else block.pair_up
    torch.nn.init.zeros_(pair_update.W_L.weight)
    torch.nn.init.zeros_(pair_update.W_R.weight)
    torch.nn.init.zeros_(block.pair_ffn.lin.weight)
    torch.nn.init.zeros_(block.pair_ffn.lin.bias)


def test_decoder_pair_update_adds_the_residual_once():
    # Regression: the decoder did P = P + pair_update(...) and P = P + pair_ffn(P)
    # although both already return P + delta, so |P| grew ~4x per layer.
    block = DecoderPairBlock(
        embedding_dim=EMBEDDING_DIMENSION,
        num_heads=2,
        dim_feedforward=16,
        d_pair=PAIR_DIMENSION,
        d_descriptor=EMBEDDING_DIMENSION,
        d_geo=RADIAL_BASIS_DIMENSION,
        dropout=0.0,
    ).eval()
    _silence_pair_deltas(block)
    atoms, pairs, radial_basis, padding_mask, pair_mask, descriptor = _block_inputs()

    with torch.no_grad():
        _, pairs_out = block(
            atoms, pairs, descriptor, radial_basis, padding_mask, pair_mask
        )

    assert (pairs_out.norm() / pairs.norm()).item() == pytest.approx(1.0, abs=1e-6)
    torch.testing.assert_close(pairs_out, pairs)


def test_encoder_pair_update_adds_the_residual_once():
    block = PairBlock(
        embedding_dim=EMBEDDING_DIMENSION,
        num_heads=2,
        d_pair=PAIR_DIMENSION,
        dim_feedforward=16,
        d_geo=RADIAL_BASIS_DIMENSION,
        dropout=0.0,
    ).eval()
    _silence_pair_deltas(block)
    atoms, pairs, radial_basis, padding_mask, pair_mask, _ = _block_inputs()

    with torch.no_grad():
        _, pairs_out = block(atoms, padding_mask, pairs, radial_basis, pair_mask)

    torch.testing.assert_close(pairs_out, pairs)


def test_encoder_pair_ffn_uses_the_configured_dropout():
    # Regression: the encoder built PairFFN(d_pair) with its default dropout 0.1
    # whatever the configured dropout was.
    block = PairBlock(
        embedding_dim=EMBEDDING_DIMENSION,
        num_heads=2,
        d_pair=PAIR_DIMENSION,
        dim_feedforward=16,
        d_geo=RADIAL_BASIS_DIMENSION,
        dropout=0.37,
    )
    assert block.pair_ffn.dropout.p == pytest.approx(0.37)
