import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from remedi_data.yaml_io import write_yaml


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
        # self.task_names: list[str] = dataset_config.get_task_names()
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

        self.grad_norm_data = []

    def __enter__(self) -> "TrainingTelemetry":
        return self

    def __exit__(self, *_exception_info):
        self.finish()

    def finish(self):
        self.dump_loss_history()
        self.dump_probe_history()

        # plot all the plots

        # self.plot_grad_norm()

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def zip_task_losses(self, task_losses: torch.Tensor):
        if task_losses is None:
            return None

        task_losses = task_losses.cpu().detach().numpy()
        zipped = dict(
            zip(
                self.task_names,
                task_losses.tolist(),
                strict=False,
            )
        )

        return zipped

    def check_best_val_epoch(self, validation_loss_current_epoch):
        if validation_loss_current_epoch < self.best_validation_loss:
            self.best_validation_loss = validation_loss_current_epoch
            self.best_epoch = True
        else:
            self.best_epoch = False

    def log_epoch(
        self,
        epoch,
        avg_train_loss,
        avg_train_loss_per_task,
        avg_validation_loss,
        avg_validation_loss_per_task,
        current_lr,
    ):
        self.check_best_val_epoch(avg_validation_loss)

        training_task_loss = self.zip_task_losses(avg_train_loss_per_task)

        validation_task_loss = self.zip_task_losses(avg_validation_loss_per_task)

        epoch_train_data = {
            "epoch": epoch + 1,
            "validation_loss": avg_validation_loss,
            "train_loss": avg_train_loss,
            "task_validation_loss": validation_task_loss,
            "task_train_loss": training_task_loss,
            "learning_rate": current_lr,
        }

        self.loss_data.append(epoch_train_data)
        self.dump_loss_history()

        print(
            f"Epoch {epoch} Training Loss: {avg_train_loss} Validation Loss: {avg_validation_loss}"
        )

        self._log_to_wandb(epoch_train_data)

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

    def get_track_grad_norm_fn(self):
        def capture_grad(grad):  # grad has same shape as M
            # L2 norm over all non-batch dims, then mean over batch
            norm = grad.flatten(1).norm(dim=1).mean()
            self.grad_norm_data.append(norm.item())

        return capture_grad

    def plot_grad_norm(self):
        fig = plt.figure()
        plt.scatter(np.arange(len(self.grad_norm_data)), self.grad_norm_data)
        fig.savefig(f"{self.out_dir}/dL_dM_grad_norm.png")
