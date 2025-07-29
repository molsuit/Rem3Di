import argparse
import os
from datetime import datetime
from pathlib import Path
import math
import numpy as np
import pydantic_yaml as pyaml
import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from threedscriptors.data_handling.indexed_subset import IndexedSubset


from threedscriptors.data_handling.dataset import RegressionDatasetwithPositions, RegressionDatasetwithRandomWalks, RegressionWithAuxAndPositionsDataset
from threedscriptors.training.dataset_splitting import DatasetSplitting, SplitConfig, SplitStrategy

import wandb
from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.data_handling.dataset_io import load_data_from_disk
from threedscriptors.data_handling.sample import sample_collate_fn
from threedscriptors.evaluation.training_evaluation import regression_pipeline, chiral_regression_pipeline
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.regression_training import multitask_masked_loss, DynamicallyWeighedMultitaskLoss, BaseMultitaskLoss
from threedscriptors.training.telemetry import TrainingTelemetry
from threedscriptors.training.data_normalization import DataNormalizationModule

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    """
    Parse command-line arguments and return the run_name.
    """
    parser = argparse.ArgumentParser(
        description="Parse the --run_name argument for naming runs"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run (e.g., experiment identifier)",
    )
    args = parser.parse_args()
    return args.run_name


run_name = parse_args()
torch.manual_seed(0)
np.random.seed(0)

training_run_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs"
)
training_idx = len(list(training_run_dir.glob("*/")))
now = datetime.now()
training_data_dir = training_run_dir / Path(
    f"{training_idx}-{now.strftime("%Y_%m_%d_%H_%M_%S")}-{run_name}"
)

os.makedirs(training_data_dir)

split_config = SplitConfig(
    strategy= SplitStrategy.SINGLE,
    N_folds = None,
    N_repeats = None,
    shuffle = True
)

training_config = TrainingConfig(
    batch_size=64,
    epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-3,
    max_grad_norm=1.0,
    wandb_active=True,
    split_config=split_config,
    training_data_dir=training_data_dir,
    mace_model_path="/share/snw30/projects/mace_model/MACE-OFF24_medium.model",
    dataset_path="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt_training",
    test_dataset_path="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/qm9_test",
    model_dir="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/cmrt_training",
    normalized_targets=True,
)

architecture_config = pyaml.parse_yaml_file_as(
    ArchitectureConfig,
    f"{training_config.model_dir}/architecture_config.yaml",
)

print("Start Dataloading")
dataset = reload_dataset_pipeline(training_config.dataset_path).build()
#dataset.expand_embedding_num_atoms(29)
dataset = dataset.convert_to_dataset_type(RegressionWithAuxAndPositionsDataset)

dataset_splitting = DatasetSplitting(dataset)

lowest_val_losses = []

for train_idx, val_idx, split_name in dataset_splitting.get_split(training_config.split_config):


    os.makedirs(f"{training_config.training_data_dir}/{split_name.lower()}")

    train_dataset = IndexedSubset(dataset, train_idx)
    valid_dataset = IndexedSubset(dataset, val_idx)



    training_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=sample_collate_fn,
    )
    validation_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=sample_collate_fn,
    )


    data_normalization = DataNormalizationModule(dataset = train_dataset)
    

    task_configs = data_normalization.task_configs

    inv_mean_per_dim, inv_std_per_dim = data_normalization.get_atomic_embedding_normalization_constants()


    mb = ModelBuilder(architecture_config=architecture_config)
    mb.insert_task_configs_into_regression_heads(task_configs)
    model = mb.build_model(
        mean_atomic_embedding=inv_mean_per_dim,
        std_atomic_embedding=inv_std_per_dim
    )


    print(f"Trainable Parameters: {mb.N_trainable_parameters}")


    config = {
        "architecture_config": architecture_config.model_dump(),
        "training_config": training_config.model_dump(),
        "dataset_config": dataset.dataset_config.model_dump(),
    }

        
    model.to(device)
    model.encoder.to(dtype=torch.float32)
    model.multitask_heads.to(dtype=torch.float32)


    loss_fn = BaseMultitaskLoss()

    #all_params = model.parameters()

    all_params = (
    list(model.encoder.parameters())
    + list(model.preprocessor.geometric_preprocessor.parameters())
)

    optimizer = optim.AdamW(
        [{"params" : all_params, "lr" : training_config.learning_rate, "weight_decay" : training_config.weight_decay},
        ]
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=[training_config.learning_rate], total_steps=training_config.epochs * len(training_loader)
    )
    best_model_path = f"{training_config.training_data_dir}/best_model.pth"

    loss_fn.to(device)
    
    
    with TrainingTelemetry(training_config=training_config, dataset_config= dataset.dataset_config, run_name = run_name, split_name=split_name, config = config) as telemetry:


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


            for batch_idx, samples in enumerate(training_loader):
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
                accumulated_train_loss_per_task += batch_weighed_loss_per_task_train.detach()


            avg_train_loss = accumulated_train_loss / (batch_idx+1)
            avg_train_loss_per_task = accumulated_train_loss_per_task / (batch_idx + 1)

            accumulated_validation_loss = 0.0
            accumulated_validation_loss_per_task = torch.zeros(
                len(dataset.dataset_config.tasks), device=device
            )
            
            loss_fn.eval()
            model.eval()

            with torch.no_grad():
                for batch_idx, val_samples in enumerate(validation_loader):
                    val_samples = data_normalization(val_samples)
                    val_samples.to_(device)

                    val_output = model(val_samples)

                    loss, batch_weighed_loss_per_task_val = loss_fn(val_samples, val_output)
                    accumulated_validation_loss_per_task += batch_weighed_loss_per_task_val

                    accumulated_validation_loss += loss.item()

                avg_validation_loss = accumulated_validation_loss / (batch_idx + 1)
                avg_validation_loss_per_task = accumulated_validation_loss_per_task / (batch_idx + 1)


                current_lr = scheduler.get_last_lr()[0]

                telemetry.log_epoch(epoch, avg_train_loss, avg_train_loss_per_task, avg_validation_loss, avg_validation_loss_per_task, current_lr)

                if telemetry.best_epoch:
                    
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  - New best model (val_loss {avg_validation_loss:.4f}), saving to {best_model_path}")


    lowest_val_losses.append(telemetry.best_validation_loss)

   
    
    print("Loading best model from", best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()



    torch.save(model.state_dict(), f"{training_config.training_data_dir}/regression_model.pth")
    
    torch.save(
        model.preprocessor.state_dict(), f"{training_config.training_data_dir}/preprocessor.pth"
    )
    torch.save(model.encoder.state_dict(), f"{training_config.training_data_dir}/encoder.pth")
    #
    pyaml.to_yaml_file(
        f"{training_config.training_data_dir}/training_config.yaml", training_config
    )


    architecture_config.reload_full_model_weights = (
        f"{training_config.training_data_dir}/regression_model.pth"
    )

    pyaml.to_yaml_file(
        f"{training_config.training_data_dir}/architecture_config.yaml", architecture_config
    )

    pyaml.to_yaml_file(
        f"{training_config.training_data_dir}/dataset_config.yaml",
        dataset.dataset_config,
    )

    figs = {}

    from threedscriptors.configuration.data_config import DatasetSplit
    training_evaluation_pipeline = regression_pipeline(train_dataset, dataset_split=DatasetSplit.TRAIN)
    training_evaluation_pipeline.evaluate(model)
    train_figs, train_result_report = training_evaluation_pipeline.output_results(
        output_directory=f"{training_config.training_data_dir}/trainset_results", model_name=run_name
    )




    validation_evaluation_pipeline = regression_pipeline(valid_dataset, dataset_split=DatasetSplit.TRAIN)
    validation_evaluation_pipeline.evaluate(model)
    train_figs, train_result_report =validation_evaluation_pipeline.output_results(
        output_directory=f"{training_config.training_data_dir}/valset_results", model_name=run_name
    )

    
    if training_config.wandb_active:
        for figname, figure in figs.items():
            wandb.log({figname: figure})

