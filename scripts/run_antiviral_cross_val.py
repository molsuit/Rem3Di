import argparse
import os
from pathlib import Path

import numpy as np
import pydantic_yaml as pyaml
import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from threedscriptors.configuration.architecture_config import (
    RegressionArchitectureConfig,
)
from threedscriptors.configuration.data_config import DatasetSplit
from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.data_build_pipeline import (
    AtomicPositionsStage,
    PipelineOrchestrator,
    ReduceConformerStage,
    ReloadFromDiskStage,
)
from threedscriptors.data_handling.dataset import (
    RegressionDatasetwithPositions,
)
from threedscriptors.data_handling.indexed_subset import IndexedSubset
from threedscriptors.data_handling.sample import sample_collate_fn
from threedscriptors.evaluation.training_evaluation import (
    regression_pipeline,
)
from threedscriptors.training.data_normalization import DataNormalizationModule
from threedscriptors.training.dataset_splitting import (
    DatasetSplitting,
    SplitConfig,
    SplitStrategy,
)
from threedscriptors.training.regression_training import (
    BaseMultitaskLoss,
)
from threedscriptors.training.telemetry import TrainingTelemetry

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """
    Parse command-line arguments and return the run_name.
    """
    parser = argparse.ArgumentParser(
        description="Parse the --run_name argument for naming runs"
    )
    parser.add_argument(
        "--n_conf",
        type=int,
        required=True,
        help="Name of the run (e.g., experiment identifier)",
    )
    args = parser.parse_args()
    return args.n_conf


N_conformers = parse_args()
torch.manual_seed(0)
np.random.seed(0)

training_run_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/0-av_potency_conformal_sampling"
)

run_name = f"{N_conformers}_conformers"
training_data_dir = training_run_dir / Path(f"{run_name}")

os.makedirs(training_data_dir, exist_ok=True)

split_config = SplitConfig(
    strategy=SplitStrategy.REPEATED_CV, N_folds=5, N_repeats=2, shuffle=True
)

training_config = TrainingConfig(
    batch_size=64,
    epochs=30,
    learning_rate=2e-5,
    weight_decay=1e-3,
    max_grad_norm=1.0,
    wandb_active=True,
    split_config=split_config,
    training_data_dir=training_data_dir,
    dataset_path="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/antiviral_potency_64_conf_full",
    model_dir="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/antiviral_potency_full",
    normalized_targets=True,
)

architecture_config = pyaml.parse_yaml_file_as(
    RegressionArchitectureConfig,
    f"{training_config.model_dir}/architecture_config.yaml",
)


print("Start Dataloading")

reload_pipeline_stages = [
    ReloadFromDiskStage(training_config.dataset_path),
    AtomicPositionsStage(),
    ReduceConformerStage(N_conformers),
]
orchestrator = PipelineOrchestrator(reload_pipeline_stages)

dataset = orchestrator.build()
dataset = dataset.convert_to_dataset_type(RegressionDatasetwithPositions)

dataset_splitting = DatasetSplitting(dataset)

lowest_val_losses = {}


patience = 7  # epochs


epochs_since_improve = 0


for train_idx, val_idx, split_name in dataset_splitting.get_split(
    training_config.split_config
):
    split_dir = f"{training_config.training_data_dir}/{split_name.lower()}"
    os.makedirs(split_dir, exist_ok=True)

    train_dataset = IndexedSubset(dataset, train_idx)
    valid_dataset = IndexedSubset(dataset, val_idx)

    training_loader = DataLoader(
        train_dataset,
        batch_size=training_config.dataloader.batch_sampling.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=sample_collate_fn,
    )
    validation_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.dataloader.batch_sampling.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=sample_collate_fn,
    )

    data_normalization = DataNormalizationModule(dataset=train_dataset)

    task_configs = data_normalization.task_configs

    inv_mean_per_dim, inv_std_per_dim = (
        data_normalization.get_atomic_embedding_normalization_constants()
    )

    architecture_config.insert_task_configs(task_configs)
    model = architecture_config.build(
        mean_atomic_embedding=inv_mean_per_dim, std_atomic_embedding=inv_std_per_dim
    )

    print(
        f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    config = {
        "architecture_config": architecture_config.model_dump(),
        "training_config": training_config.model_dump(),
        "dataset_config": dataset.dataset_config.model_dump(),
    }

    model.to(device)
    model.encoder.to(dtype=torch.float32)
    model.multitask_heads.to(dtype=torch.float32)

    loss_fn = BaseMultitaskLoss()

    all_params = model.parameters()

    #    all_params = (
    #    list(model.encoder.parameters())
    #    + list(model.preprocessor.geometric_preprocessor.parameters())
    # )
    # all_params= model.multitask_heads.parameters()

    optimizer = optim.AdamW(
        [
            {
                "params": all_params,
                "lr": training_config.learning_rate,
                "weight_decay": training_config.weight_decay,
            },
        ]
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[training_config.learning_rate],
        total_steps=training_config.epochs * len(training_loader),
    )
    best_model_path = f"{split_dir}/best_model.pth"

    loss_fn.to(device)

    with TrainingTelemetry(
        training_config=training_config,
        dataset_config=dataset.dataset_config,
        run_name=run_name,
        split_name=split_name,
        config=config,
    ) as telemetry:
        print("Starting Training")

        for epoch in range(training_config.epochs):
            # Initialize task and total train losses
            accumulated_train_loss = 0.0
            accumulated_train_loss_per_task = torch.zeros(
                len(dataset.dataset_config.tasks), device=device
            )

            loss_fn.train()
            model.train()
            optimizer.zero_grad()

            for samples in training_loader:
                samples = data_normalization(samples)
                samples.to_(device)
                model_output = model(samples)

                loss, batch_weighed_loss_per_task_train = loss_fn(samples, model_output)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=training_config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                accumulated_train_loss += loss.item()
                accumulated_train_loss_per_task += (
                    batch_weighed_loss_per_task_train.detach()
                )

            avg_train_loss = accumulated_train_loss / len(training_loader)
            avg_train_loss_per_task = accumulated_train_loss_per_task / len(
                training_loader
            )

            accumulated_validation_loss = 0.0
            accumulated_validation_loss_per_task = torch.zeros(
                len(dataset.dataset_config.tasks), device=device
            )

            loss_fn.eval()
            model.eval()

            with torch.no_grad():
                for val_samples in validation_loader:
                    val_samples = data_normalization(val_samples)
                    val_samples.to_(device)

                    val_output = model(val_samples)

                    loss, batch_weighed_loss_per_task_val = loss_fn(
                        val_samples, val_output
                    )
                    accumulated_validation_loss_per_task += (
                        batch_weighed_loss_per_task_val
                    )

                    accumulated_validation_loss += loss.item()

                avg_validation_loss = accumulated_validation_loss / len(
                    validation_loader
                )
                avg_validation_loss_per_task = (
                    accumulated_validation_loss_per_task / len(validation_loader)
                )

                current_lr = scheduler.get_last_lr()[0]

                telemetry.log_epoch(
                    epoch,
                    avg_train_loss,
                    avg_train_loss_per_task,
                    avg_validation_loss,
                    avg_validation_loss_per_task,
                    current_lr,
                )

                if telemetry.best_epoch:
                    best_val = avg_validation_loss
                    epochs_since_improve = 0
                    torch.save(model.state_dict(), best_model_path)
                    print(
                        f"  - New best model (val_loss {avg_validation_loss:.6f}), saved to {best_model_path}"
                    )

                else:
                    epochs_since_improve += 1
                    print(
                        f"  - No improvement ({epochs_since_improve}/{patience}) | best={best_val:.6f}"
                    )

                if epochs_since_improve >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Best val_loss={best_val:.6f}."
                    )
                    break

    lowest_val_losses[split_name] = telemetry.best_validation_loss

    print("Loading best model from", best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    torch.save(model.state_dict(), f"{split_dir}/regression_model.pth")

    torch.save(
        model.preprocessor.state_dict(),
        f"{split_dir}/preprocessor.pth",
    )
    torch.save(model.encoder.state_dict(), f"{split_dir}/encoder.pth")
    #
    pyaml.to_yaml_file(f"{split_dir}/training_config.yaml", training_config)

    pyaml.to_yaml_file(
        f"{split_dir}/architecture_config.yaml",
        architecture_config,
    )

    pyaml.to_yaml_file(
        f"{split_dir}/dataset_config.yaml",
        dataset.dataset_config,
    )

    figs = {}

    training_evaluation_pipeline = regression_pipeline(
        train_dataset, dataset_split=DatasetSplit.TRAIN
    )
    training_evaluation_pipeline.evaluate(model)
    train_figs, train_result_report = training_evaluation_pipeline.output_results(
        output_directory=f"{split_dir}/trainset_results",
        model_name=run_name,
    )

    validation_evaluation_pipeline = regression_pipeline(
        valid_dataset, dataset_split=DatasetSplit.VALIDATION
    )
    validation_evaluation_pipeline.evaluate(model)
    train_figs, train_result_report = validation_evaluation_pipeline.output_results(
        output_directory=f"{split_dir}/valset_results",
        model_name=run_name,
    )

    if training_config.wandb_active:
        for figname, figure in figs.items():
            wandb.log({figname: figure})


# Write lowest validation losses to file
with open(f"{training_data_dir}/lowest_val_losses.txt", "w") as f:
    for split_name, loss in lowest_val_losses.items():
        f.write(f"{split_name}: {loss:.4f}\n")
