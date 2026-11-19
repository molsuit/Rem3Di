"""The fixed projection of pooled descriptors (standardise, then principal
components) and its embedding config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from remedi.embedding import MaceMeanConfig, MaceMeanProjectionConfig
from remedi.embedding.projection import PooledProjection, fit_pooled_projection
from remedi.model.checkpoint import ARCHITECTURE_FILENAME, ENCODER_WEIGHT_FILES


def _values(count: int = 500) -> np.ndarray:
    generator = np.random.default_rng(0)
    latent = generator.normal(size=(count, 3))
    mixing = generator.normal(size=(3, 12))
    scales = np.logspace(-2, 3, 12)  # very different channel scales
    return (latent @ mixing + 0.01 * generator.normal(size=(count, 12))) * scales


def test_the_projection_finds_the_latent_dimensions_regardless_of_scale() -> None:
    projection = fit_pooled_projection(_values(), dimension=4)
    assert projection.components.shape == (4, 12)
    assert projection.explained_variance_ratio[:3].sum() > 0.99
    projected = projection.apply(_values())
    assert projected.shape == (500, 4) and projected.dtype == np.float32
    # Principal axes: uncorrelated projections with decreasing variance.
    covariance = np.cov(projected, rowvar=False)
    assert np.allclose(covariance - np.diag(np.diag(covariance)), 0.0, atol=1e-3)
    assert np.all(np.diff(np.diag(covariance)) <= 1e-6)


def test_a_projection_round_trips_through_its_file(tmp_path: Path) -> None:
    projection = fit_pooled_projection(_values(), dimension=2)
    path = tmp_path / "projection.npz"
    projection.save(path)
    loaded = PooledProjection.load(path)
    np.testing.assert_array_equal(loaded.apply(_values()), projection.apply(_values()))


def test_fitting_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="components"):
        fit_pooled_projection(_values(5), dimension=6)
    values = _values()
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_pooled_projection(values, dimension=2)


def test_the_projection_identity_covers_the_projection_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    (model_dir / ARCHITECTURE_FILENAME).write_text("mace_config: {model_path: null}\n")
    for filename in ENCODER_WEIGHT_FILES:
        (model_dir / filename).write_bytes(filename.encode())
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    fit_pooled_projection(_values(), dimension=2).save(first)
    fit_pooled_projection(_values(), dimension=3).save(second)

    def identity(path: Path) -> str:
        return MaceMeanProjectionConfig(
            name="mace_pca", model_dir=model_dir, projection_path=path
        ).identity()

    assert identity(first) != identity(second)
    assert identity(first) != MaceMeanConfig(name="m", model_dir=model_dir).identity()
