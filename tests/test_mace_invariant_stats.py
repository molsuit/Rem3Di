"""Unit tests for the streaming-stats helpers used by the MACE invariant
analysis script."""

import numpy as np
import pytest
import torch
from e3nn.o3 import Irreps

from remedi.evaluation.descriptor_analysis.mace_invariant_stats import (
    _CovarianceAccumulator,
    _ReservoirSampler,
    _Welford,
    coding_rate,
    get_layer_invariant_specs,
    mean_abs_correlation,
    participation_ratio,
    per_dim_entropy,
)


def test_get_layer_invariant_specs_two_mace_layers():
    """MACE-POLAR-1-M signature: 2 layers, each emitting 512x0e + 512x1o."""
    irreps = Irreps("512x0e+512x1o+512x0e+512x1o")
    specs = get_layer_invariant_specs(irreps)
    assert len(specs) == 2
    assert (specs[0].layer_index, specs[0].start, specs[0].stop) == (0, 0, 512)
    assert (specs[1].layer_index, specs[1].start, specs[1].stop) == (1, 512, 1024)
    assert specs[0].width == 512
    assert specs[1].width == 512


def test_get_layer_invariant_specs_uneven_widths():
    irreps = Irreps("4x0e+2x1o+8x0e")
    specs = get_layer_invariant_specs(irreps)
    assert [(s.layer_index, s.start, s.stop) for s in specs] == [(0, 0, 4), (1, 4, 12)]


def test_get_layer_invariant_specs_no_invariants_raises():
    with pytest.raises(ValueError):
        get_layer_invariant_specs(Irreps("3x1o"))


def test_welford_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=2.0, scale=3.0, size=(5000, 7)).astype(np.float64)
    w = _Welford(dim=7, device=torch.device("cpu"))
    for chunk in np.array_split(x, 11):
        w.update(torch.from_numpy(chunk))
    np.testing.assert_allclose(w.mean.numpy(), x.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(w.std.numpy(), x.std(axis=0, ddof=1), atol=1e-7)
    np.testing.assert_allclose(w.min.numpy(), x.min(axis=0))
    np.testing.assert_allclose(w.max.numpy(), x.max(axis=0))


def test_covariance_accumulator_matches_numpy():
    rng = np.random.default_rng(1)
    d = 6
    x = rng.normal(size=(2000, d)).astype(np.float64)
    acc = _CovarianceAccumulator(dim=d, device=torch.device("cpu"))
    for chunk in np.array_split(x, 8):
        acc.update(torch.from_numpy(chunk))
    cov = acc.covariance().numpy()
    np.testing.assert_allclose(cov, np.cov(x, rowvar=False), atol=1e-9)


def test_reservoir_sampler_unbiased_size():
    sampler = _ReservoirSampler(capacity=200, dim=3, seed=0)
    rng = np.random.default_rng(0)
    total = 0
    for _ in range(20):
        n = int(rng.integers(50, 200))
        sampler.update(rng.normal(size=(n, 3)).astype(np.float32))
        total += n
    sample = sampler.sample()
    assert sample.shape == (200, 3)
    assert sampler.seen == total


def test_participation_ratio_uniform_spectrum_recovers_dim():
    eig = np.ones(8)
    assert participation_ratio(eig) == pytest.approx(8.0, rel=1e-6)


def test_participation_ratio_one_hot_is_one():
    eig = np.array([1.0, 0.0, 0.0, 0.0])
    assert participation_ratio(eig) == pytest.approx(1.0, rel=1e-6)


def test_coding_rate_zero_for_zero_cov():
    cov = np.zeros((4, 4))
    assert coding_rate(cov, eps_sq=1.0, n_samples=100) == pytest.approx(0.0, abs=1e-9)


def test_coding_rate_increases_with_variance():
    cov_small = 0.01 * np.eye(4)
    cov_large = 1.0 * np.eye(4)
    r_small = coding_rate(cov_small, eps_sq=0.25, n_samples=1000)
    r_large = coding_rate(cov_large, eps_sq=0.25, n_samples=1000)
    assert r_large > r_small


def test_mean_abs_correlation_uncorrelated_is_small():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5000, 6))
    cov = np.cov(x, rowvar=False)
    assert mean_abs_correlation(cov) < 0.05


def test_mean_abs_correlation_perfect_correlation_is_one():
    # Two perfectly correlated copies of one signal
    rng = np.random.default_rng(3)
    z = rng.normal(size=(2000, 1))
    x = np.hstack([z, z])
    cov = np.cov(x, rowvar=False)
    assert mean_abs_correlation(cov) == pytest.approx(1.0, abs=1e-6)


def test_per_dim_entropy_uniform_close_to_max():
    rng = np.random.default_rng(4)
    x = rng.uniform(size=(20000, 3)).astype(np.float64)
    H, H_norm = per_dim_entropy(
        x,
        bins=64,
        range_min=np.zeros(3),
        range_max=np.ones(3),
    )
    # Each dim should approach log2(64) = 6 bits; allow small slack.
    assert np.all(H > 5.7)
    assert np.all(H_norm > 0.95)


def test_per_dim_entropy_constant_dim_is_zero():
    x = np.zeros((1000, 2))
    x[:, 1] = np.linspace(0, 1, 1000)
    H, _ = per_dim_entropy(
        x,
        bins=32,
        range_min=np.array([0.0, 0.0]),
        range_max=np.array([0.0, 1.0]),
    )
    assert H[0] == 0.0
    assert H[1] > 4.5
