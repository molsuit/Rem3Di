"""Frozen diagnostic linear probes on the molecular descriptor.

A fixed held-out probe set (M validation molecules) is re-embedded every N steps
under ``eval`` / ``no_grad`` and a closed-form ridge is fit on a fixed fit/score
split to measure how linearly-decodable the physicochemical target columns are.
The probe never touches gradients or the optimizer — it is pure measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from remedi.data_handling.sample import Sample


def _ridge_predict(
    x_fit: np.ndarray, y_fit: np.ndarray, x_score: np.ndarray, alpha: float
) -> np.ndarray:
    """Closed-form ridge predictions. Primal when n>=d, dual otherwise.

    Both forms are algebraically identical for ridge regression; we pick the
    cheaper system to solve (``d x d`` primal vs ``n x n`` dual).
    """
    n, d = x_fit.shape
    if n >= d:
        a = x_fit.T @ x_fit + alpha * np.eye(d)
        w = np.linalg.solve(a, x_fit.T @ y_fit)
        return x_score @ w
    g = x_fit @ x_fit.T + alpha * np.eye(n)
    coeffs = np.linalg.solve(g, y_fit)
    return (x_score @ x_fit.T) @ coeffs


@dataclass
class ProbeResult:
    """Per-property probe scores for one probe run."""

    r2: dict[str, float] = field(default_factory=dict)
    mae: dict[str, float] = field(default_factory=dict)
    accuracy: dict[str, float] = field(default_factory=dict)
    macro_r2: float = float("nan")

    def to_metrics(self) -> dict[str, float]:
        """Flatten to dotted keys for ``TrainingTelemetry.log_metrics``."""
        out: dict[str, float] = {}
        for name, v in self.r2.items():
            out[f"probe/{name}/r2"] = v
        for name, v in self.mae.items():
            out[f"probe/{name}/mae"] = v
        for name, v in self.accuracy.items():
            out[f"probe/{name}/acc"] = v
        out["probe/macro_r2"] = self.macro_r2
        return out


class LinearProbeMonitor:
    """Fit-and-score ridge probes on a fixed held-out molecule set."""

    def __init__(
        self,
        *,
        probe_samples: list[Sample],
        labels: np.ndarray,
        masks: np.ndarray,
        property_names: Sequence[str],
        collate_fn: Callable[[list[Sample]], Sample],
        device: torch.device | str,
        every_n_steps: int = 500,
        ridge_alpha: float = 1.0,
        val_fraction: float = 0.3,
        batch_size: int = 256,
        report_count_accuracy: bool = False,
        seed: int = 0,
    ) -> None:
        n = len(probe_samples)
        if n < 4:
            raise ValueError(f"probe set too small ({n}); need >= 4 molecules")
        if labels.shape[0] != n or masks.shape[0] != n:
            raise ValueError("labels/masks rows must match number of probe samples")

        self.probe_samples = probe_samples
        self.labels = np.asarray(labels, dtype=np.float64)
        self.masks = np.asarray(masks).astype(bool)
        self.property_names = list(property_names)
        self.collate_fn = collate_fn
        self.device = torch.device(device)
        self.every_n_steps = int(every_n_steps)
        self.ridge_alpha = float(ridge_alpha)
        self.batch_size = int(batch_size)
        self.report_count_accuracy = report_count_accuracy

        # Fixed fit/score row split (disjoint, both held out from training).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_score = max(2, round(n * val_fraction))
        n_score = min(n_score, n - 2)  # leave >=2 for fitting
        self.score_idx = np.sort(perm[:n_score])
        self.fit_idx = np.sort(perm[n_score:])

    def maybe_run(self, global_step: int, preprocessor, encoder) -> dict[str, float]:
        """Return probe metrics on cadence, else ``{}`` (no embedding cost)."""
        if self.every_n_steps <= 0 or global_step % self.every_n_steps != 0:
            return {}
        x = self._embed(preprocessor, encoder)
        return self._fit_score(x).to_metrics()

    def _embed(self, preprocessor, encoder) -> np.ndarray:
        """Embed the fixed probe set in eval/no_grad; restore prior train modes."""
        pre_training = preprocessor.training
        enc_training = encoder.training
        preprocessor.eval()
        encoder.eval()
        parts: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for start in range(0, len(self.probe_samples), self.batch_size):
                    group = self.probe_samples[start : start + self.batch_size]
                    # Re-collate every run: the preprocessor mutates the batch
                    # Sample in place, but the raw per-molecule Samples are
                    # untouched, so a fresh collate yields a clean batch.
                    sample = self.collate_fn(group)
                    sample.to_(self.device)
                    descriptor = encoder(preprocessor(sample))
                    parts.append(
                        descriptor.flat.detach().to("cpu", torch.float64).numpy()
                    )
        finally:
            if pre_training:
                preprocessor.train()
            if enc_training:
                encoder.train()
        return np.concatenate(parts, axis=0)

    def _fit_score(self, x: np.ndarray) -> ProbeResult:
        # Standardize features on the (full) fit set; reused for every column.
        x_fit_all = x[self.fit_idx]
        mu = x_fit_all.mean(axis=0)
        sd = x_fit_all.std(axis=0)
        sd[sd == 0.0] = 1.0
        xs = (x - mu) / sd

        result = ProbeResult()
        r2_values: list[float] = []
        for col, name in enumerate(self.property_names):
            fit_rows = self.fit_idx[self.masks[self.fit_idx, col]]
            score_rows = self.score_idx[self.masks[self.score_idx, col]]
            if fit_rows.size < 2 or score_rows.size < 2:
                continue

            y_fit = self.labels[fit_rows, col]
            y_true = self.labels[score_rows, col]
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
            if ss_tot == 0.0:
                continue  # constant target -> R^2 undefined

            y_mu = y_fit.mean()
            y_sd = y_fit.std() or 1.0
            pred = (
                _ridge_predict(
                    xs[fit_rows],
                    (y_fit - y_mu) / y_sd,
                    xs[score_rows],
                    self.ridge_alpha,
                )
                * y_sd
                + y_mu
            )

            ss_res = float(np.sum((y_true - pred) ** 2))
            r2 = 1.0 - ss_res / ss_tot
            result.r2[name] = r2
            result.mae[name] = float(np.mean(np.abs(y_true - pred)))
            r2_values.append(r2)

            if self.report_count_accuracy and np.allclose(y_true, np.round(y_true)):
                result.accuracy[name] = float(
                    np.mean(np.round(pred) == np.round(y_true))
                )

        result.macro_r2 = float(np.mean(r2_values)) if r2_values else float("nan")
        return result
