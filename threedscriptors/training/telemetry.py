import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml


class TrainingTelemetry:
    def __init__(
        self, wandb_active: bool, run_name: str, group_name: str, out_dir: Path, config
    ):
        # self.task_names: list[str] = dataset_config.get_task_names()
        self.wandb_active = wandb_active
        self.out_dir = out_dir

        self.loss_data = []

        self.best_validation_loss = math.inf
        self.best_epoch = True

        if self.wandb_active:
            self._run = wandb.init(
                project="threedscriptors",
                entity="threedscriptors",
                name=run_name,
                group=group_name,
                config=config,
            )

        self.grad_norm_data = []

    def __enter__(self) -> "TrainingTelemetry":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()

    def finish(self):
        self.dump_loss_history()

        # plot all the plots

        # self.plot_grad_norm()

        if self.wandb_active:
            wandb.finish()

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

        print(
            f"Epoch {epoch} Training Loss: {avg_train_loss} Validation Loss: {avg_validation_loss}"
        )

        if self.wandb_active:
            wandb.log(epoch_train_data)

    def log_pretraining_epoch(self, epoch, train_loss, validation_loss, current_lr):
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

        self.loss_data.append(epoch_data)

        if self.wandb_active:
            wandb.log(data=epoch_data, step=epoch)

    def dump_loss_history(self):
        with open(
            f"{self.out_dir}/training_losses.yaml",
            "x",
        ) as f:
            yaml.safe_dump(self.loss_data, f)

    def get_track_grad_norm_fn(self):
        def capture_grad(grad):  # grad has same shape as M
            # L2 norm over all non‑batch dims, then mean over batch
            norm = grad.flatten(1).norm(dim=1).mean()
            self.grad_norm_data.append(norm.item())

        return capture_grad

    def plot_grad_norm(self):
        fig = plt.figure()
        plt.scatter(np.arange(len(self.grad_norm_data)), self.grad_norm_data)
        fig.savefig(f"{self.out_dir}/dL_dM_grad_norm.png")
