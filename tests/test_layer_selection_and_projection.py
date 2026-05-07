"""Tests for the per-layer selection + invariant projection extension to
``EmbeddingPreprocessConfig`` / ``AtomicDescriptorPreprocessor``.

These tests intentionally avoid loading a real MACE checkpoint: they construct
a preprocess config with explicit per-layer irreps, mirroring what the cascade
would populate from ``MaceConfig.get_per_layer_irreps()``.
"""

from __future__ import annotations

import pydantic_yaml as pyaml
import pytest
import torch
from e3nn.o3 import Irreps

from threedscriptors.configuration.architecture_config import (
    EmbeddingPreprocessConfig,
    IdentityInvariantProjectionConfig,
    LinearInvariantProjectionConfig,
    MLPInvariantProjectionConfig,
    PrecomputedInvariantNormalizationConfig,
)

# Two-layer MACE-style stack: each layer emits 8 invariant + 4 vector channels.
LAYER_IRREPS_STR = "8x0e+4x1o"
PER_LAYER_IRREPS = [Irreps(LAYER_IRREPS_STR), Irreps(LAYER_IRREPS_STR)]
FULL_INPUT_IRREPS = PER_LAYER_IRREPS[0] + PER_LAYER_IRREPS[1]


def _make_config(
    *,
    layer_indices: list[int] | None,
    projection,
    pseudoscalars: bool = False,
):
    return EmbeddingPreprocessConfig(
        input_irreps=FULL_INPUT_IRREPS,
        mace_per_layer_irreps=PER_LAYER_IRREPS,
        mace_layer_indices=layer_indices,
        pseudoscalar_dimension=0,
        chiral_embedding_dimension=0,
        pseudoscalars=pseudoscalars,
        invariant_normalization_config=PrecomputedInvariantNormalizationConfig(),
        invariant_projection_config=projection,
    )


# --------------------------------------------------------------------------- #
# Computed fields                                                             #
# --------------------------------------------------------------------------- #


def test_selected_input_irreps_all_layers_default():
    cfg = _make_config(
        layer_indices=None,
        projection=IdentityInvariantProjectionConfig(),
    )
    assert cfg.selected_input_irreps == FULL_INPUT_IRREPS
    assert cfg.selected_channel_indices is None
    assert cfg.selected_invariant_dimension == 16  # 8 + 8


@pytest.mark.parametrize(
    "layer_indices, expected_channels",
    [
        ([0], list(range(0, 20))),  # first layer's 20 channels
        ([1], list(range(20, 40))),  # second layer's 20 channels
        ([0, 1], None),  # both layers => no slicing needed
    ],
)
def test_selected_channel_indices(layer_indices, expected_channels):
    cfg = _make_config(
        layer_indices=layer_indices,
        projection=IdentityInvariantProjectionConfig(),
    )
    assert cfg.selected_channel_indices == expected_channels


@pytest.mark.parametrize(
    "layer_indices, expected_inv_dim",
    [
        (None, 16),
        ([0], 8),
        ([1], 8),
        ([0, 1], 16),
    ],
)
def test_selected_invariant_dimension(layer_indices, expected_inv_dim):
    cfg = _make_config(
        layer_indices=layer_indices,
        projection=IdentityInvariantProjectionConfig(),
    )
    assert cfg.selected_invariant_dimension == expected_inv_dim


def test_invalid_layer_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        _make_config(
            layer_indices=[5],
            projection=IdentityInvariantProjectionConfig(),
        )


def test_unsorted_layer_indices_raises():
    with pytest.raises(ValueError, match="sorted ascending"):
        _make_config(
            layer_indices=[1, 0],
            projection=IdentityInvariantProjectionConfig(),
        )


# --------------------------------------------------------------------------- #
# Output dim reflects the projection                                          #
# --------------------------------------------------------------------------- #


def test_output_dim_identity_no_projection():
    cfg = _make_config(
        layer_indices=None,
        projection=IdentityInvariantProjectionConfig(),
    )
    assert cfg.projected_invariant_dimension == cfg.selected_invariant_dimension
    assert cfg.output_irreps_dim == cfg.selected_invariant_dimension


def test_output_dim_linear_projection():
    cfg = _make_config(
        layer_indices=None,
        projection=LinearInvariantProjectionConfig(output_dim=4),
    )
    assert cfg.projected_invariant_dimension == 4
    assert cfg.output_irreps_dim == 4


def test_output_dim_mlp_projection():
    cfg = _make_config(
        layer_indices=[0],
        projection=MLPInvariantProjectionConfig(
            output_dim=6, hidden_dimensions=[12, 6]
        ),
    )
    # First layer alone has 8 invariants; MLP projects to 6.
    assert cfg.selected_invariant_dimension == 8
    assert cfg.projected_invariant_dimension == 6
    assert cfg.output_irreps_dim == 6


# --------------------------------------------------------------------------- #
# End-to-end forward                                                          #
# --------------------------------------------------------------------------- #


def _random_embedding(batch: int, n_atoms: int) -> torch.Tensor:
    return torch.randn(batch, n_atoms, FULL_INPUT_IRREPS.dim, dtype=torch.float64)


@pytest.mark.parametrize("layer_indices", [None, [0], [1], [0, 1]])
@pytest.mark.parametrize(
    "projection",
    [
        IdentityInvariantProjectionConfig(),
        LinearInvariantProjectionConfig(output_dim=5),
        MLPInvariantProjectionConfig(output_dim=7, hidden_dimensions=[16]),
    ],
)
def test_atomic_preprocessor_forward(layer_indices, projection):
    cfg = _make_config(layer_indices=layer_indices, projection=projection)
    pre = cfg.build().double()

    # Pretend stats were precomputed for the selected invariants.
    sel_dim = cfg.selected_invariant_dimension
    pre.invariant_normalization.set_stats(
        mean=torch.zeros(sel_dim), std=torch.ones(sel_dim), overwrite=True
    )

    embeddings = _random_embedding(batch=2, n_atoms=3)
    padding_mask = torch.zeros((2, 3), dtype=torch.bool)
    out = pre(embeddings, padding_mask)
    assert out.preprocessed_atomic_embeddings.shape == (2, 3, cfg.output_irreps_dim)
    assert out.preprocessed_atomic_embeddings.dtype == torch.float32


def test_atomic_preprocessor_layer_slice_matches_manual():
    """Forward output of a layer-1-only preprocessor must equal the output of a
    no-selection preprocessor fed only the layer-1 channel slice."""
    torch.manual_seed(0)

    layer_idx = 1

    cfg_sliced = _make_config(
        layer_indices=[layer_idx],
        projection=IdentityInvariantProjectionConfig(),
    )
    pre_sliced = cfg_sliced.build()
    sel_dim = cfg_sliced.selected_invariant_dimension
    pre_sliced.invariant_normalization.set_stats(
        mean=torch.zeros(sel_dim), std=torch.ones(sel_dim), overwrite=True
    )

    cfg_manual = EmbeddingPreprocessConfig(
        # Only layer 1's irreps -> the "post-slice" view.
        input_irreps=PER_LAYER_IRREPS[layer_idx],
        mace_per_layer_irreps=[PER_LAYER_IRREPS[layer_idx]],
        pseudoscalar_dimension=0,
        chiral_embedding_dimension=0,
        pseudoscalars=False,
        invariant_projection_config=IdentityInvariantProjectionConfig(),
    )
    pre_manual = cfg_manual.build()
    pre_manual.invariant_normalization.set_stats(
        mean=torch.zeros(sel_dim), std=torch.ones(sel_dim), overwrite=True
    )

    embeddings = _random_embedding(batch=2, n_atoms=4)
    padding = torch.zeros((2, 4), dtype=torch.bool)

    out_sliced = pre_sliced(embeddings, padding).preprocessed_atomic_embeddings
    layer_dim = PER_LAYER_IRREPS[layer_idx].dim
    start = sum(p.dim for p in PER_LAYER_IRREPS[:layer_idx])
    out_manual = pre_manual(
        embeddings[..., start : start + layer_dim], padding
    ).preprocessed_atomic_embeddings

    torch.testing.assert_close(out_sliced, out_manual)


def test_stats_autoslice_from_full_invariants():
    """build() should accept stats sized for the full invariants and slice
    them down to the selected layer's channels."""
    cfg = _make_config(
        layer_indices=[0],
        projection=IdentityInvariantProjectionConfig(),
    )
    full_inv_dim = cfg.invariant_irreps.dim  # 16 (both layers)
    sel_inv_dim = cfg.selected_invariant_dimension  # 8 (first layer)
    assert full_inv_dim == 16 and sel_inv_dim == 8

    mean = torch.arange(full_inv_dim, dtype=torch.float64)
    std = torch.ones(full_inv_dim, dtype=torch.float64) * 2.0
    pre = cfg.build(mean_atomic_embedding=mean, std_atomic_embedding=std)

    # Layer 0 invariant channels are the first 8.
    expected_mean = mean[:sel_inv_dim].view(1, 1, sel_inv_dim)
    expected_std = std[:sel_inv_dim].view(1, 1, sel_inv_dim)
    torch.testing.assert_close(
        pre.invariant_normalization.mean,
        expected_mean.to(pre.invariant_normalization.mean.dtype),
    )
    torch.testing.assert_close(
        pre.invariant_normalization.std,
        expected_std.to(pre.invariant_normalization.std.dtype),
    )


# --------------------------------------------------------------------------- #
# YAML roundtrip                                                              #
# --------------------------------------------------------------------------- #


def test_yaml_roundtrip_with_projection():
    cfg = _make_config(
        layer_indices=[0],
        projection=MLPInvariantProjectionConfig(
            output_dim=6, hidden_dimensions=[16, 8], dropout=0.1
        ),
    )
    config_str = pyaml.to_yaml_str(cfg)
    reconstructed = pyaml.parse_yaml_raw_as(EmbeddingPreprocessConfig, config_str)
    assert pyaml.to_yaml_str(reconstructed) == config_str
