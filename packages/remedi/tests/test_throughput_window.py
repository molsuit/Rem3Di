"""The throughput window splits wall time into loader wait and the rest."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from remedi.training.telemetry import (
    THROUGHPUT_FILENAME,
    ThroughputWindow,
    TrainingTelemetry,
)


def test_the_window_separates_loader_wait_from_compute() -> None:
    window = ThroughputWindow("cpu")
    for _ in range(3):
        time.sleep(0.02)  # waiting for the loader
        window.batch_ready()
        time.sleep(0.01)  # the training step
        window.step_done(atom_count=100)
    metrics = window.metrics()
    assert metrics["atoms_per_s"] == pytest.approx(300 / 0.09, rel=0.5)
    assert metrics["data_wait_ms"] == pytest.approx(20.0, rel=0.5)
    assert 0.4 < metrics["data_wait_fraction"] < 0.9
    assert "gpu_utilization_percent" not in metrics  # CPU: no GPU metrics


def test_metrics_start_a_new_window() -> None:
    window = ThroughputWindow("cpu")
    window.batch_ready()
    window.step_done(atom_count=5)
    window.metrics()
    assert (window.steps, window.atoms, window.data_wait_seconds) == (0, 0, 0.0)


def test_throughput_records_are_appended_as_json_lines(tmp_path: Path) -> None:
    telemetry = TrainingTelemetry(
        wandb_active=False,
        run_name="run",
        group_name="group",
        out_dir=tmp_path,
        config=None,
    )
    telemetry.log_throughput({"global_step": 50, "step_ms": 1.5})
    telemetry.log_throughput({"global_step": 100, "step_ms": 2.5})
    lines = (tmp_path / THROUGHPUT_FILENAME).read_text().splitlines()
    assert [json.loads(line)["global_step"] for line in lines] == [50, 100]
