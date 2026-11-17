"""The pair-distance preprocessor must not depend on the batch's padded size."""

from __future__ import annotations

import pytest
import torch

from remedi.configuration.architecture_config import (
    RelativeDistancePositionalEncodingConfig,
)
from remedi.training.data.sample import Sample


@pytest.mark.parametrize("coordinate_offset", [0.0, 10.0, 50.0])
def test_pair_encoding_independent_of_padding(coordinate_offset: float):
    # Regression: torch.cdist switches to a matmul formulation above 25 atoms;
    # in float32 that changed distances by up to 0.04 Å at a 50 Å offset, so a
    # molecule's encoding depended on how far its batch was padded.
    geometric = RelativeDistancePositionalEncodingConfig(
        N_radial_basis_functions=16, distance_cutoff=6.0, d_projection=8
    ).build()
    generator = torch.Generator().manual_seed(0)
    number_of_atoms = 6
    positions = (
        torch.randn(number_of_atoms, 3, generator=generator) * 1.5 + coordinate_offset
    )

    def encode(padded_atoms: int):
        padded = torch.zeros(1, padded_atoms, 3)
        padded[0, :number_of_atoms] = positions
        padding_mask = torch.arange(padded_atoms).unsqueeze(0) >= number_of_atoms
        with torch.no_grad():
            pair, radial_basis, _ = geometric(
                Sample(atomic_positions=padded, padding_mask=padding_mask)
            )
        return (
            pair[0, :number_of_atoms, :number_of_atoms],
            radial_basis[0, :number_of_atoms, :number_of_atoms],
        )

    pair_alone, radial_basis_alone = encode(number_of_atoms)
    pair_padded, radial_basis_padded = encode(40)

    torch.testing.assert_close(radial_basis_padded, radial_basis_alone)
    torch.testing.assert_close(pair_padded, pair_alone)
