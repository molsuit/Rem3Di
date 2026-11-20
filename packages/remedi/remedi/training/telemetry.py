import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

from remedi_data.yaml_io import write_yaml

#: Per-window throughput records, one JSON object per line.
THROUGHPUT_FILENAME = "throughput.jsonl"


def gpu_utilization_percent(device: torch.device | str) -> float | None:
    """The GPU's utilisation over the driver's last sample period (NVML), or
    None off CUDA or when NVML is unavailable."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    try:
        return float(torch.cuda.utilization(device))
    except Exception:  # NVML missing or not permitted: telemetry is optional
        return None


class ForwardTimer:
    """The time spent in a module's forward calls, for a breakdown of the step.

    On CUDA the calls are bracketed by CUDA events, so timing adds no host
    synchronisation; the events are read in :meth:`take_seconds`. An interval runs
    from the call's start to the GPU finishing the work it queued, so it also counts
    the GPU waiting for the host to launch that work. On the CPU it is host time.
    """

    def __init__(self, module: nn.Module, device: torch.device | str):
        self.uses_cuda_events = torch.device(device).type == "cuda"
        self._intervals: list[tuple[Any, Any]] = []
        self._started: Any = None
        self._hooks = [
            module.register_forward_pre_hook(self._start),
            module.register_forward_hook(self._stop),
        ]

    def _mark(self) -> Any:
        if not self.uses_cuda_events:
            return perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _start(self, module: nn.Module, inputs: Any) -> None:
        self._started = self._mark()

    def _stop(self, module: nn.Module, inputs: Any, output: Any) -> None:
        if self._started is not None:
            self._intervals.append((self._started, self._mark()))
            self._started = None

    def take_seconds(self) -> tuple[float, int]:
        """The total seconds and number of the calls since the last take (or
        :meth:`reset`); starts counting anew."""
        intervals, self._intervals = self._intervals, []
        if not intervals:
            return 0.0, 0
        if self.uses_cuda_events:
            intervals[-1][1].synchronize()
            seconds = sum(start.elapsed_time(end) for start, end in intervals) / 1000.0
        else:
            seconds = sum(end - start for start, end in intervals)
        return seconds, len(intervals)

    def reset(self) -> None:
        self._intervals = []
        self._started = None

    def remove(self) -> None:
        """Detach from the module."""
        for hook in self._hooks:
            hook.remove()
        self.reset()


@dataclass
class ThroughputWindow:
    """Wall time of a window of training steps, split into waiting for the
    loader and everything else.

    The training step synchronises with the GPU every step (the finiteness
    checks of loss and gradients), so host-side times are GPU times. Call
    :meth:`batch_ready` when the loader has handed over a batch and
    :meth:`step_done` when the optimiser step is finished. ``forward_timers``
    add, per name, the milliseconds per step spent in that module's forward and
    their share of the window's wall time (``<name>_ms``, ``<name>_fraction``).
    """

    device: torch.device | str
    forward_timers: dict[str, ForwardTimer] = field(default_factory=dict)
    started: float = field(default_factory=perf_counter)
    last_step_end: float = field(default_factory=perf_counter)
    steps: int = 0
    atoms: int = 0
    data_wait_seconds: float = 0.0

    def __post_init__(self) -> None:
        self._reset_peak_memory()
        for timer in self.forward_timers.values():
            timer.reset()

    def batch_ready(self) -> None:
        self.data_wait_seconds += perf_counter() - self.last_step_end

    def step_done(self, atom_count: int) -> None:
        self.steps += 1
        self.atoms += atom_count
        self.last_step_end = perf_counter()

    def metrics(self) -> dict[str, float]:
        """The window's metrics; then starts the next window."""
        seconds = max(perf_counter() - self.started, 1e-9)
        steps = max(self.steps, 1)
        result: dict[str, float] = {
            "atoms_per_s": self.atoms / seconds,
            "step_ms": 1000.0 * seconds / steps,
            "data_wait_ms": 1000.0 * self.data_wait_seconds / steps,
            "data_wait_fraction": self.data_wait_seconds / seconds,
        }
        for name, timer in self.forward_timers.items():
            timed_seconds, _ = timer.take_seconds()
            result[f"{name}_ms"] = 1000.0 * timed_seconds / steps
            result[f"{name}_fraction"] = timed_seconds / seconds
        utilization = gpu_utilization_percent(self.device)
        if utilization is not None:
            result["gpu_utilization_percent"] = utilization
        if torch.device(self.device).type == "cuda":
            result["gpu_peak_memory_gib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
            )
        self.started = self.last_step_end = perf_counter()
        self.steps = self.atoms = 0
        self.data_wait_seconds = 0.0
        self._reset_peak_memory()
        return result

    def _reset_peak_memory(self) -> None:
        if torch.device(self.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


def _start_wandb_run(run_name: str, group_name: str, config: Any) -> Any:
    """A Weights & Biases run (the ``telemetry`` extra).

    Raises:
        ImportError: naming the extra, if wandb is not installed.
    """
    try:
        import wandb
    except ImportError as error:
        raise ImportError(
            f"wandb_active needs wandb ({error}): pip install 'remedi[telemetry]'"
        ) from error
    return wandb.init(
        project="remedi",
        entity="remedi",
        name=run_name,
        group=group_name,
        config=config,
    )


class TrainingTelemetry:
    def __init__(
        self, wandb_active: bool, run_name: str, group_name: str, out_dir: Path, config
    ):
        self.out_dir = out_dir

        self.loss_data = []
        # Mid-epoch linear-probe trajectory, dumped to probe_history.yaml so the
        # per-property R²/MAE survive as a parseable artifact (not just wandb).
        self.probe_data: list[dict] = []

        self.best_validation_loss = math.inf
        self.best_epoch = True

        #: The Weights & Biases run, ``None`` when wandb logging is off.
        self._wandb_run = (
            _start_wandb_run(run_name, group_name, config) if wandb_active else None
        )

    def __enter__(self) -> "TrainingTelemetry":
        return self

    def __exit__(self, *_exception_info):
        self.finish()

    def finish(self):
        self.dump_loss_history()
        self.dump_probe_history()

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def state_dict(self) -> dict[str, Any]:
        """What a resumed run needs to continue the histories and the
        best-epoch selection."""
        return {
            "loss_data": self.loss_data,
            "probe_data": self.probe_data,
            "best_validation_loss": self.best_validation_loss,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loss_data = list(state["loss_data"])
        self.probe_data = list(state["probe_data"])
        self.best_validation_loss = float(state["best_validation_loss"])

    def check_best_val_epoch(self, validation_loss_current_epoch):
        if validation_loss_current_epoch < self.best_validation_loss:
            self.best_validation_loss = validation_loss_current_epoch
            self.best_epoch = True
        else:
            self.best_epoch = False

    def log_pretraining_epoch(
        self,
        epoch,
        train_loss,
        validation_loss,
        current_lr,
        extra_metrics: dict[str, float] | None = None,
    ):
        self.check_best_val_epoch(validation_loss)

        print(
            f"Epoch {epoch} Training Loss: {train_loss} Validation Loss: {validation_loss}"
        )

        epoch_data = {
            "denoising_train_loss": train_loss,
            "denoising_val_loss": validation_loss,
            "epoch": epoch + 1,
            "learning_rate": current_lr,
        }
        if extra_metrics:
            epoch_data.update(extra_metrics)

        self.loss_data.append(epoch_data)
        self.dump_loss_history()

        self._log_to_wandb(epoch_data)

    def log_epoch_metrics(self, metrics: dict[str, float]) -> None:
        """Record one epoch's metrics: written to ``training_losses.yaml`` now
        and logged to wandb."""
        self.loss_data.append(dict(metrics))
        self.dump_loss_history()
        self._log_to_wandb(metrics)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Mid-epoch metric flush (e.g. rolling throughput).

        Auto-steps via wandb's internal counter so it stays monotonic
        alongside epoch-level logs that don't pin step explicitly.
        """
        self._log_to_wandb(metrics)

    def log_throughput(self, record: dict[str, float]) -> None:
        """Append one throughput window to ``throughput.jsonl`` and log it to wandb."""
        with open(Path(self.out_dir) / THROUGHPUT_FILENAME, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        self._log_to_wandb(record)

    def log_probe(self, global_step: int, metrics: dict[str, float]) -> None:
        """Record one linear-probe point: persisted to disk AND logged to wandb.

        Unlike ``log_metrics`` (wandb-only throughput flush), the probe
        trajectory is accumulated into ``probe_data`` so it is dumped to
        ``probe_history.yaml`` as a first-class, parseable artifact.
        """
        record = {"global_step": int(global_step), **metrics}
        self.probe_data.append(record)
        self.dump_probe_history()
        self._log_to_wandb(record)

    def _log_to_wandb(self, data: dict[str, Any]) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(data)

    def dump_loss_history(self):
        """Rewrite ``training_losses.yaml``; called after every epoch so a job
        killed by its time limit keeps the epochs it finished."""
        write_yaml(Path(self.out_dir) / "training_losses.yaml", self.loss_data)

    def dump_probe_history(self):
        if not self.probe_data:
            return
        write_yaml(Path(self.out_dir) / "probe_history.yaml", self.probe_data)
