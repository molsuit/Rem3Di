"""The pair block shared by the encoder and the decoder."""

from __future__ import annotations

import pytest
import torch

from remedi.model.pair_biased_attention import PairOuterProdUpdate
from remedi.model.pair_block import PairBlock, PairTransition

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


def _block(*, decoder: bool, update_pair: bool = True, dropout: float = 0.0):
    return PairBlock(
        embedding_dim=EMBEDDING_DIMENSION,
        num_heads=2,
        d_pair=PAIR_DIMENSION,
        dim_feedforward=16,
        d_geo=RADIAL_BASIS_DIMENSION,
        dropout=dropout,
        descriptor_dimension=EMBEDDING_DIMENSION if decoder else None,
        update_pair=update_pair,
    )


def _run(block: PairBlock, *, decoder: bool):
    atoms, pairs, radial_basis, padding_mask, pair_mask, descriptor = _block_inputs()
    with torch.no_grad():
        return block(
            atoms,
            padding_mask,
            pairs,
            radial_basis,
            pair_mask,
            descriptor_tokens=descriptor if decoder else None,
        )


@pytest.mark.parametrize("decoder", [False, True])
def test_the_pair_update_adds_the_residual_once(decoder: bool) -> None:
    # Regression: the decoder did P = P + pair_update(...) and P = P + pair_ffn(P)
    # although both already return P + delta, so |P| grew ~4x per layer.
    block = _block(decoder=decoder).eval()
    torch.nn.init.zeros_(block.pair_update.W_L.weight)
    torch.nn.init.zeros_(block.pair_update.W_R.weight)
    pairs = _block_inputs()[1]

    _, pairs_out = _run(block, decoder=decoder)

    torch.testing.assert_close(pairs_out, pairs)


def test_the_pair_transition_is_the_identity_at_initialisation() -> None:
    # Regression: the old gated pair FFN claimed "no effect at start" but its
    # zero-initialised gate was sigmoid(0) = 0.5.
    transition = PairTransition(PAIR_DIMENSION, dropout=0.0).eval()
    pairs = _block_inputs()[1]
    with torch.no_grad():
        torch.testing.assert_close(transition(pairs), pairs)


def test_the_pair_transition_uses_the_configured_dropout() -> None:
    assert _block(decoder=False, dropout=0.37).pair_transition.dropout.p == (
        pytest.approx(0.37)
    )


def test_the_outer_product_update_does_not_grow_with_the_atom_norm() -> None:
    """The atoms are layer-normalised before the (quadratic) outer product."""
    torch.manual_seed(0)
    update = PairOuterProdUpdate(
        EMBEDDING_DIMENSION, PAIR_DIMENSION, RADIAL_BASIS_DIMENSION
    )
    atoms, pairs, radial_basis, _, pair_mask, _ = _block_inputs()
    pairs = torch.zeros_like(pairs)
    with torch.no_grad():
        small = update(atoms, pairs, radial_basis, pair_mask)
        large = update(100.0 * atoms, pairs, radial_basis, pair_mask)
    assert small.abs().max() > 0
    torch.testing.assert_close(large, small, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(small, small.transpose(1, 2))


@pytest.mark.parametrize("decoder", [False, True])
def test_a_block_without_pair_update_returns_the_same_atoms(decoder: bool) -> None:
    """Skipping the pair update leaves the atom output unchanged and passes the
    pair tensor through."""
    pairs = _block_inputs()[1]
    outputs = []
    for update_pair in (True, False):
        torch.manual_seed(0)
        block = _block(decoder=decoder, update_pair=update_pair).eval()
        outputs.append(_run(block, decoder=decoder))
    (atoms_updated, _), (atoms_skipped, pairs_skipped) = outputs
    torch.testing.assert_close(atoms_skipped, atoms_updated)
    torch.testing.assert_close(pairs_skipped, pairs)
    assert block.pair_update is None and block.pair_transition is None


def test_a_decoder_block_needs_the_descriptor_tokens() -> None:
    block = _block(decoder=True).eval()
    with pytest.raises(ValueError, match="descriptor_tokens"):
        _run(block, decoder=False)
