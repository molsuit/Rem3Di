"""Tests for the frozen linear-probe monitor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from remedi.data_handling.sample import Sample, yield_molecules_collate_fn
from remedi.training.linear_probe import LinearProbeMonitor, _ridge_predict


def _make_monitor(labels, masks, *, every_n_steps=1, val_fraction=0.3):
    m = labels.shape[0]
    samples = [
        Sample(
            atomic_positions=torch.zeros((2, 3)),
            atomic_numbers=torch.tensor([6, 1]),
            total_charge=torch.tensor(0.0),
            multiplicity=torch.tensor(1.0),
        )
        for _ in range(m)
    ]
    names = [f"p{i}" for i in range(labels.shape[1])]
    return LinearProbeMonitor(
        probe_samples=samples,
        labels=labels,
        masks=masks,
        property_names=names,
        collate_fn=yield_molecules_collate_fn,
        device="cpu",
        every_n_steps=every_n_steps,
        ridge_alpha=1.0,
        val_fraction=val_fraction,
        batch_size=m,
        seed=0,
    )


def test_ridge_primal_dual_agree():
    rng = np.random.default_rng(0)
    # n < d so _ridge_predict uses the dual form; compare against the primal.
    x_fit = rng.standard_normal((20, 50))
    y_fit = rng.standard_normal(20)
    x_score = rng.standard_normal((7, 50))
    alpha = 1.0
    dual = _ridge_predict(x_fit, y_fit, x_score, alpha)
    w = np.linalg.solve(x_fit.T @ x_fit + alpha * np.eye(50), x_fit.T @ y_fit)
    primal = x_score @ w
    np.testing.assert_allclose(dual, primal, rtol=1e-6, atol=1e-6)


def test_probe_recovers_linear_signal():
    rng = np.random.default_rng(1)
    m, d, p = 200, 8, 3
    x = rng.standard_normal((m, d))
    w = rng.standard_normal((d, p))
    y_linear = x @ w + 0.01 * rng.standard_normal((m, p))
    y_random = rng.standard_normal((m, p))
    masks = np.ones((m, p), dtype=bool)

    monitor = _make_monitor(y_linear, masks)
    res = monitor._fit_score(x)
    assert min(res.r2.values()) > 0.9
    assert res.macro_r2 > 0.9

    monitor_rand = _make_monitor(y_random, masks)
    res_rand = monitor_rand._fit_score(x)
    assert max(res_rand.r2.values()) < 0.5


def test_fully_masked_column_is_omitted():
    rng = np.random.default_rng(2)
    m, d, p = 100, 5, 3
    x = rng.standard_normal((m, d))
    y = x @ rng.standard_normal((d, p))
    masks = np.ones((m, p), dtype=bool)
    masks[:, 1] = False  # column 1 fully masked

    monitor = _make_monitor(y, masks)
    res = monitor._fit_score(x)  # must not raise
    assert "p1" not in res.r2
    assert "p0" in res.r2 and "p2" in res.r2


def test_maybe_run_respects_cadence_and_restores_train_mode():
    rng = np.random.default_rng(3)
    m, p = 8, 2
    labels = rng.standard_normal((m, p))
    masks = np.ones((m, p), dtype=bool)
    monitor = _make_monitor(labels, masks, every_n_steps=10)

    class RaisingEnc(nn.Module):
        def forward(self, _):  # should never be called off-cadence
            raise AssertionError("encoder called off cadence")

    pre = nn.Identity()
    pre.train()
    # Off cadence: returns {} without embedding.
    assert monitor.maybe_run(7, pre, RaisingEnc()) == {}

    class FakeEnc(nn.Module):
        def forward(self, sample):
            b = sample.total_charge.shape[0]
            return SimpleNamespace(flat=torch.randn(b, 4))

    enc = FakeEnc()
    pre.train()
    enc.train()
    metrics = monitor.maybe_run(10, pre, enc)
    assert isinstance(metrics, dict)
    assert "probe/macro_r2" in metrics
    # eval() during embed must be reverted.
    assert pre.training and enc.training
