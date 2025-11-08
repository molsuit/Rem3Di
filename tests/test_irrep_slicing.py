import pytest
from e3nn.o3 import Irreps

from threedscriptors.configuration.architecture_config import (
    EmbeddingPreprocessConfig,
)
from threedscriptors.utils.model_utils import get_pseudoscalar_indices


@pytest.mark.parametrize(
    "irreps_str, expected_idx",
    [
        # one pseudoscalar block (l = 0, p = -1)
        ("2x0o + 1x0e + 3x1o", [0]),

        # no pseudoscalars at all
        ("1x0e + 2x1o", []),

        # two separate pseudoscalar blocks
        ("1x0o + 1x1o + 1x0o", [0, 2]),
    ],
)
def test_get_pseudoscalar_indices(irreps_str, expected_idx):
    """
    The function must return the exact slices corresponding to every
    pseudoscalar (l = 0, odd parity) block, in order.
    """
    irreps = Irreps(irreps_str)

    # What we expect: slices for the indices we marked as pseudoscalar
    all_slices = irreps.slices()
    expected_slices = [all_slices[i] for i in expected_idx]

    result = get_pseudoscalar_indices(irreps)

    assert len(result) == len(expected_slices)

    assert result == expected_slices

    expected_dim = sum(s.stop - s.start for s in expected_slices)
    result_dim   = sum(s.stop - s.start for s in result)
    assert expected_dim == result_dim



def test_equivariant_dimension_scalar_only():
    """
    If the input contains only scalars (l = 0), the equivariant
    dimension should be zero.
    """
    cfg = EmbeddingPreprocessConfig(
        input_irreps=Irreps("4x0e"),          # four even scalars
        pseudoscalar_dimension=1              # required field
    )

    assert cfg.input_equivariant_dimension == 0


def test_equivariant_dimension_mixed_l():
    """
    For mixed irreps, equivariant_dimension must equal the total
    dimension carried by all blocks with l > 0.
    """
    irreps = Irreps("3x0e + 2x1o + 1x2e")      # 3 scalars, 2 vectors, 1 quadrupole
    expected = Irreps("2x1o + 1x2e").dim       # 2x3  + 1x5 = 11

    cfg = EmbeddingPreprocessConfig(
        input_irreps=irreps,
        pseudoscalar_dimension=1
    )

    assert cfg.input_equivariant_dimension == expected



def make_cfg(irreps: str | Irreps) -> EmbeddingPreprocessConfig:
    """
    Minimal config factory so we don't repeat boiler-plate.
    Only the two mandatory arguments are set; everything
    else falls back to the Pydantic defaults.
    """
    return EmbeddingPreprocessConfig(
        input_irreps=Irreps(irreps),
        pseudoscalar_dimension=1,      # <- required by the model
    )



def test_input_dimension_matches_irreps_dim():
    """
    The field `input_dimension` must equal Irreps.dim for *any* input.
    """
    irr = Irreps("2x0e + 3x1o + 1x2e")   # total dim = 2·1 + 3·3 + 1·5 = 16
    cfg = make_cfg(irr)
    assert cfg.input_dimension == irr.dim



@pytest.mark.parametrize(
    "irreps_str, expected_dim",
    [
        ("4x0e", 4),                      # only even scalars
        ("2x0e + 1x0o", 3),               # even + odd scalars (pseudoscalars)
        ("3x1o + 1x2e", 0),               # no l == 0 blocks
        ("2x0e + 3x1o + 1x2e", 2),        # mixed: only 2x0e contribute
    ],
)
def test_input_invariant_dimension(irreps_str, expected_dim):
    """
    The invariant dimension is the total size of l == 0 blocks,
    irrespective of parity.
    """
    cfg = make_cfg(irreps_str)
    assert cfg.input_invariant_dimension == expected_dim
