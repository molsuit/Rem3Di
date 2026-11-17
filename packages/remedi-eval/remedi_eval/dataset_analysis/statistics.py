"""Summary statistics and timing shared by the dataset analyses."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)


class DistributionStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p05: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    max: float | None = None

    @classmethod
    def from_array(cls, x: np.ndarray) -> DistributionStats:
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            return cls(n=0)
        q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        return cls(
            n=int(x.size),
            mean=float(x.mean()),
            std=float(x.std()),
            min=float(x.min()),
            p05=float(q[0]),
            p25=float(q[1]),
            p50=float(q[2]),
            p75=float(q[3]),
            p95=float(q[4]),
            max=float(x.max()),
        )


@contextmanager
def timed(name: str, timings: dict[str, float]) -> Generator[None]:
    """Record the wall-clock seconds of the ``with`` body in ``timings[name]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
        log.info("%s took %.2fs", name, elapsed)
