import pytest
from e3nn.o3 import Irreps

from remedi.configuration.architecture_config import (
    EmbeddingPreprocessConfig,
)
from remedi.utils.model_utils import get_pseudoscalar_indices


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
    result_dim = sum(s.stop - s.start for s in result)
    assert expected_dim == result_dim


def test_input_irreps_concatenates_the_layers():
    cfg = EmbeddingPreprocessConfig(
        mace_per_layer_irreps=["2x0e + 3x1o", "1x0e + 1x2e"],
        pseudoscalar_dimension=1,
        chiral_embedding_dimension=1,
    )
    assert cfg.input_irreps == Irreps("2x0e + 3x1o + 1x0e + 1x2e")
    assert cfg.invariant_irreps == Irreps("2x0e + 1x0e")
