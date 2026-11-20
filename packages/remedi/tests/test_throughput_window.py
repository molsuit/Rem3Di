"""The throughput window splits wall time into loader wait and the rest."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from torch import nn

from remedi.training.telemetry import (
    THROUGHPUT_FILENAME,
    ForwardTimer,
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


class _Sleeper(nn.Module):
    def __init__(self, seconds: float):
        super().__init__()
        self.seconds = seconds

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        time.sleep(self.seconds)
        return value


def test_forward_timer_counts_the_calls_since_the_last_take() -> None:
    module = _Sleeper(0.01)
    timer = ForwardTimer(module, "cpu")
    module(torch.zeros(1))
    module(torch.zeros(1))
    seconds, calls = timer.take_seconds()
    assert calls == 2
    assert seconds == pytest.approx(0.02, rel=0.5)
    assert timer.take_seconds() == (0.0, 0)
    timer.remove()
    module(torch.zeros(1))
    assert timer.take_seconds() == (0.0, 0)


def test_the_window_reports_the_share_of_timed_modules() -> None:
    module = _Sleeper(0.02)
    timer = ForwardTimer(module, "cpu")
    module(torch.zeros(1))  # before the window: not counted
    window = ThroughputWindow("cpu", {"featurizer": timer})
    for _ in range(2):
        window.batch_ready()
        module(torch.zeros(1))
        time.sleep(0.02)  # the rest of the step
        window.step_done(atom_count=10)
    metrics = window.metrics()
    assert metrics["featurizer_ms"] == pytest.approx(20.0, rel=0.5)
    assert 0.3 < metrics["featurizer_fraction"] < 0.7
