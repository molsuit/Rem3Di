"""The resumable training state and the atomic checkpoint writes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

from remedi.model.checkpoint import atomic_torch_save
from remedi.training.data.samplers import BucketBatchSampler
from remedi.training.telemetry import TrainingTelemetry
from remedi.training.training_state import (
    TRAINING_STATE_FILENAME,
    load_training_state,
    save_training_state,
)


class _Run:
    """A tiny training run: two modules, optimizer, scheduler, telemetry."""

    def __init__(self, directory: Path, seed: int) -> None:
        torch.manual_seed(seed)
        self.modules: dict[str, nn.Module] = {
            "encoder": nn.Linear(3, 3),
            "decoder": nn.Linear(3, 3),
        }
        parameters = [p for m in self.modules.values() for p in m.parameters()]
        self.optimizer = torch.optim.AdamW(parameters, lr=1e-2)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=1e-2, total_steps=20)
        self.telemetry = TrainingTelemetry(
            wandb_active=False,
            run_name="run",
            group_name="group",
            out_dir=directory,
            config={},
        )

    def train_step(self) -> float:
        inputs = torch.randn(4, 3)
        loss = self.modules["decoder"](self.modules["encoder"](inputs)).square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def save(self, directory: Path, epochs: int, steps: int) -> None:
        save_training_state(
            directory,
            completed_epochs=epochs,
            global_step=steps,
            modules=self.modules,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            telemetry=self.telemetry,
        )

    def load(self, directory: Path) -> tuple[int, int] | None:
        return load_training_state(
            directory,
            modules=self.modules,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            telemetry=self.telemetry,
        )


def test_a_fresh_directory_has_no_state(tmp_path: Path) -> None:
    assert _Run(tmp_path, seed=0).load(tmp_path) is None


def test_a_resumed_run_continues_exactly_like_the_uninterrupted_one(
    tmp_path: Path,
) -> None:
    uninterrupted = _Run(tmp_path, seed=0)
    for _ in range(3):
        uninterrupted.train_step()
    uninterrupted.telemetry.log_pretraining_epoch(0, 1.0, 0.5, current_lr=1e-3)
    uninterrupted.save(tmp_path, epochs=1, steps=3)
    expected = [uninterrupted.train_step() for _ in range(3)]

    resumed = _Run(tmp_path, seed=123)  # different weights and random stream
    assert resumed.load(tmp_path) == (1, 3)
    assert [resumed.train_step() for _ in range(3)] == expected
    assert resumed.scheduler.get_last_lr() == uninterrupted.scheduler.get_last_lr()
    assert resumed.telemetry.best_validation_loss == 0.5
    assert resumed.telemetry.loss_data[0]["denoising_val_loss"] == 0.5


def test_a_state_of_another_architecture_is_refused(tmp_path: Path) -> None:
    run = _Run(tmp_path, seed=0)
    run.save(tmp_path, epochs=1, steps=1)
    other = _Run(tmp_path, seed=0)
    other.modules["decoder"] = nn.Linear(3, 5)
    with pytest.raises(RuntimeError, match="size mismatch"):
        other.load(tmp_path)
    other.modules["head"] = nn.Linear(3, 3)
    other.modules["decoder"] = nn.Linear(3, 3)
    with pytest.raises(KeyError, match="head"):
        other.load(tmp_path)


def test_an_interrupted_save_keeps_the_previous_file(tmp_path: Path, monkeypatch):
    path = tmp_path / TRAINING_STATE_FILENAME
    atomic_torch_save({"value": 1}, path)

    def killed(_content, _path) -> None:
        Path(_path).write_bytes(b"half a file")
        raise KeyboardInterrupt

    monkeypatch.setattr(torch, "save", killed)
    with pytest.raises(KeyboardInterrupt):
        atomic_torch_save({"value": 2}, path)
    monkeypatch.undo()
    assert torch.load(path, weights_only=True) == {"value": 1}


def test_set_epoch_reproduces_an_epochs_batches() -> None:
    lengths = [3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7]
    sampler = BucketBatchSampler(lengths, max_atoms_per_batch=10, seed=0)
    epochs = [list(sampler) for _ in range(3)]
    assert epochs[0] != epochs[1]

    resumed = BucketBatchSampler(lengths, max_atoms_per_batch=10, seed=0)
    list(resumed)  # an iteration that was not a training epoch
    resumed.set_epoch(2)
    assert list(resumed) == epochs[2]
