"""Frozen diagnostic linear probes on the molecular descriptor.

A fixed held-out probe set (M validation molecules) is re-embedded every N steps
under ``eval`` / ``no_grad`` and a ridge regression on standardised features is
fit on a fixed fit/score split to measure how linearly-decodable the
physicochemical target columns are.
The probe never touches gradients or the optimizer — it is pure measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from remedi.training.data.sample import Sample


@dataclass
class ProbeResult:
    """Per-property probe scores for one probe run."""

    r2: dict[str, float] = field(default_factory=dict)
    mae: dict[str, float] = field(default_factory=dict)
    accuracy: dict[str, float] = field(default_factory=dict)
    macro_r2: float = float("nan")

    def to_metrics(self) -> dict[str, float]:
        """Flatten to dotted keys for ``TrainingTelemetry.log_probe``."""
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
        """Fit and score one ridge per property on its labelled rows.

        Features are standardised over the property's labelled fit rows (a
        constant feature is left unscaled). Ridge predictions are linear in the
        target, so the target needs no standardisation.
        """
        result = ProbeResult()
        r2_values: list[float] = []
        for col, name in enumerate(self.property_names):
            fit_rows = self.fit_idx[self.masks[self.fit_idx, col]]
            score_rows = self.score_idx[self.masks[self.score_idx, col]]
            if fit_rows.size < 2 or score_rows.size < 2:
                continue

            y_true = self.labels[score_rows, col]
            if np.all(y_true == y_true[0]):
                continue  # constant target -> R^2 undefined

            model = make_pipeline(StandardScaler(), Ridge(alpha=self.ridge_alpha))
            model.fit(x[fit_rows], self.labels[fit_rows, col])
            prediction = model.predict(x[score_rows])

            r2 = float(r2_score(y_true, prediction))
            result.r2[name] = r2
            result.mae[name] = float(mean_absolute_error(y_true, prediction))
            r2_values.append(r2)

            if self.report_count_accuracy and np.allclose(y_true, np.round(y_true)):
                result.accuracy[name] = float(
                    np.mean(np.round(prediction) == np.round(y_true))
                )

        result.macro_r2 = float(np.mean(r2_values)) if r2_values else float("nan")
        return result
