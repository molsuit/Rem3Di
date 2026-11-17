"""The resumable state of a training run: ``training_state.pth`` in the run directory.

Written after every epoch (atomically), it holds everything a resubmitted job
needs to continue where the previous one stopped: the weights of every trained
module (the decoder included, unlike the descriptor checkpoint), the optimizer,
the scheduler, the telemetry histories and the random number generators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from remedi.model.checkpoint import atomic_torch_save
from remedi.training.telemetry import TrainingTelemetry

TRAINING_STATE_FILENAME = "training_state.pth"


def save_training_state(
    directory: Path,
    *,
    completed_epochs: int,
    global_step: int,
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    telemetry: TrainingTelemetry,
) -> None:
    """Write the state after ``completed_epochs`` epochs into ``directory``."""
    state: dict[str, Any] = {
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "telemetry": telemetry.state_dict(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    atomic_torch_save(state, Path(directory) / TRAINING_STATE_FILENAME)


def load_training_state(
    directory: Path,
    *,
    modules: dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    telemetry: TrainingTelemetry,
    map_location: str | torch.device | None = None,
) -> tuple[int, int] | None:
    """Restore a run from ``directory``; ``(completed_epochs, global_step)``.

    Returns None, leaving everything untouched, when the directory holds no
    training state (a fresh run).

    Raises:
        KeyError: if the state lacks one of ``modules``.
        RuntimeError: if a module's weights do not fit (the architecture changed).
    """
    path = Path(directory) / TRAINING_STATE_FILENAME
    if not path.is_file():
        return None
    # Optimizer and scheduler states are plain containers, not only tensors.
    state = torch.load(path, map_location=map_location, weights_only=False)
    for name, module in modules.items():
        module.load_state_dict(state["modules"][name])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    telemetry.load_state_dict(state["telemetry"])
    torch.set_rng_state(state["torch_random_state"])
    if state["cuda_random_states"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_random_states"])
    return int(state["completed_epochs"]), int(state["global_step"])
