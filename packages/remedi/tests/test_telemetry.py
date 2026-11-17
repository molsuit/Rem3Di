"""Training telemetry: the loss and probe histories survive a killed job."""

from __future__ import annotations

from pathlib import Path

import yaml

from remedi.training.telemetry import TrainingTelemetry


def _telemetry(out_dir: Path) -> TrainingTelemetry:
    return TrainingTelemetry(
        wandb_active=False,
        run_name="run",
        group_name="group",
        out_dir=out_dir,
        config={},
    )


def test_the_loss_history_is_on_disk_after_every_epoch(tmp_path: Path) -> None:
    """A SLURM timeout kills the job before ``finish``: each epoch must already
    be written."""
    telemetry = _telemetry(tmp_path)
    telemetry.log_pretraining_epoch(0, 1.0, 2.0, current_lr=1e-3)
    history = yaml.safe_load((tmp_path / "training_losses.yaml").read_text())
    assert [record["epoch"] for record in history] == [1]

    telemetry.log_pretraining_epoch(1, 0.5, 1.5, current_lr=1e-4)
    history = yaml.safe_load((tmp_path / "training_losses.yaml").read_text())
    assert [record["denoising_val_loss"] for record in history] == [2.0, 1.5]


def test_the_probe_history_is_on_disk_after_every_probe(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path)
    telemetry.log_probe(100, {"probe/macro_r2": 0.5})
    history = yaml.safe_load((tmp_path / "probe_history.yaml").read_text())
    assert history == [{"global_step": 100, "probe/macro_r2": 0.5}]


def test_finishing_after_logging_keeps_the_histories(tmp_path: Path) -> None:
    with _telemetry(tmp_path) as telemetry:
        telemetry.log_pretraining_epoch(0, 1.0, 2.0, current_lr=1e-3)
        telemetry.log_probe(10, {"probe/macro_r2": 0.1})
    assert len(yaml.safe_load((tmp_path / "training_losses.yaml").read_text())) == 1
    assert len(yaml.safe_load((tmp_path / "probe_history.yaml").read_text())) == 1


def test_epoch_metrics_are_on_disk_after_every_epoch(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path)
    telemetry.log_epoch_metrics({"epoch": 0, "val_balanced_accuracy": 0.5})
    history = yaml.safe_load((tmp_path / "training_losses.yaml").read_text())
    assert history == [{"epoch": 0, "val_balanced_accuracy": 0.5}]


def test_wandb_logging_without_the_telemetry_extra_names_it(
    tmp_path: Path, monkeypatch
) -> None:
    import sys

    import pytest

    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"remedi\[telemetry\]"):
        TrainingTelemetry(
            wandb_active=True,
            run_name="run",
            group_name="group",
            out_dir=tmp_path,
            config={},
        )
