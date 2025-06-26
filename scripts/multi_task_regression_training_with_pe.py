import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pydantic_yaml as pyaml
import torch
import yaml
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import wandb
from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.data_handling.sample import sample_collate_fn
from threedscriptors.evaluation.training_evaluation import regression_pipeline
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.regression_training import multitask_masked_loss

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

training_run_dir = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs")
training_idx = len(list(training_run_dir.glob('*/')))
now = datetime.now()
training_data_dir = training_run_dir / Path(f"{training_idx}-{now.strftime("%Y_%m_%d_%H_%M_%S")}-{run_name}")

os.makedirs(training_data_dir)

training_config = TrainingConfig(
    batch_size=64,
    epochs=20,
    learning_rate=1e-5,
    weight_decay= 1e-3,
    max_grad_norm=1.0,
    wandb_active=True,
    training_data_dir= training_data_dir,
    mace_model_path="/share/snw30/projects/mace_model/MACE-OFF24_medium.model",
    train_dataset_path="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt_train",
    validation_dataset_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt_valid",
    model_dir="/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/cmrt",
    normalized_targets=True,
)

architecture_config = pyaml.parse_yaml_file_as(
    ArchitectureConfig,
    f"{training_config.model_dir}/architecture_config.yaml",
)

train_pipeline_orchestrator = reload_dataset_pipeline(
    training_config.train_dataset_path,
)
train_dataset = train_pipeline_orchestrator.build()
mean_embeddings, std_embeddings, equivariant_scale_factor = train_dataset.get_atomic_embedding_normalization_constants(input_irreps = architecture_config.embedding_preprocess_config.input_irreps)


training_config.total_steps = len(train_dataset) * training_config.epochs


mean_target_per_task, std_target_per_task = train_dataset.dataset_config.get_mean_std_per_task()


valid_pipeline_orchestrator = reload_dataset_pipeline(
    training_config.validation_dataset_path,
    mean_targets= mean_target_per_task, std_targets= std_target_per_task
)
valid_dataset = valid_pipeline_orchestrator.build()

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
    collate_fn=sample_collate_fn,
)



mb = ModelBuilder(architecture_config=architecture_config)
model = mb.build_model(
    mean_atomic_embedding=mean_embeddings, std_atomic_embedding=std_embeddings, #equivariant_scale_factor= equivariant_scale_factor
)


print(f"Trainable Parameters: {mb.N_trainable_parameters}")


config={
            "architecture_config": architecture_config.model_dump(),
            "training_config": training_config.model_dump(),
            "train_dataset_config": train_dataset.dataset_config.model_dump(),
            "valid_dataset_config": valid_dataset.dataset_config.model_dump()
        }



if training_config.wandb_active:
    wandb.init(
        project="threedscriptors",
        entity="threedscriptors",
        name=run_name,
        config=config
    )

    wandb.watch(models = model.preprocessor, log ="all", log_freq=10)


model.to(device)


num_opt_steps = training_config.epochs * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
scheduler = OneCycleLR(
    optimizer, max_lr=training_config.learning_rate, total_steps=num_opt_steps
)

task_names = train_dataset.dataset_config.get_task_name_set()
_, stds_per_task = train_dataset.dataset_config.get_mean_std_per_task()
assert model.multitask_heads.task_heads.keys() == stds_per_task.keys()

stds = np.array(list(std_target_per_task.values()))

print("Starting Training")

loss_data = []

for epoch in range(training_config.epochs):
    running_tloss = 0.0
    weighed_loss_per_task_train = torch.zeros(
        len(train_dataset.dataset_config.tasks), device=device
    )
    model.train()
    optimizer.zero_grad()

    for _batch, samples in enumerate(training_loader):
        samples.to_(device)

        samples.padding_mask = samples.padding_mask.bool()
        model_output = model(samples)


        loss, batch_weighed_loss_per_task_train = multitask_masked_loss(
            predictions=model_output.regression_predictions,
            labels=samples.regression_targets,
            regression_mask=samples.regression_masks,
        )





        weighed_loss_per_task_train += batch_weighed_loss_per_task_train.detach()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=training_config.max_grad_norm
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_tloss += loss.item()

    avg_tloss = (
        running_tloss / (_batch + 1)
    )  # TODO: if you don't have anything else to do, you could use TorchMetrics to calculate the loss. It's a bit of a hassle to set up though
    weighed_loss_per_task_train = weighed_loss_per_task_train / (_batch + 1)

    running_vloss = 0.0
    weighed_loss_per_task_val = torch.zeros(
        len(train_dataset.dataset_config.tasks), device=device
    )

    model.eval()
    with torch.no_grad():
        for _batch, val_samples in enumerate(validation_loader):

            val_samples.to_(device)
            val_samples.padding_mask = val_samples.padding_mask.bool()

            val_output = model(val_samples)

            loss, batch_weighed_loss_per_task_val = multitask_masked_loss(
            predictions=val_output.regression_predictions,
            labels=val_samples.regression_targets,
            regression_mask=val_samples.regression_masks,
        )
            weighed_loss_per_task_val += batch_weighed_loss_per_task_val

            running_vloss += loss.item()

        avg_vloss = running_vloss / (_batch + 1)
        weighed_loss_per_task_val = weighed_loss_per_task_val / (_batch + 1)

        print(f"Epoch {epoch} Training Loss: {avg_tloss} Validation Loss: {avg_vloss}")

        current_lr = scheduler.get_last_lr()

        destandardized_loss_per_task_val = weighed_loss_per_task_val.cpu().detach().numpy() * stds

        val_dict = dict(
                zip(
                    task_names,
                    destandardized_loss_per_task_val.tolist(),
                    strict=False,
                )
            )


        destandardized_loss_per_task_train = weighed_loss_per_task_train.cpu().detach().numpy() * stds

        train_dict = dict(
                zip(
                    task_names,
                    destandardized_loss_per_task_train.tolist(),
                    strict=False,
                )
            )

        epoch_loss_dict  = {
                    "epoch" : epoch,
                    "validation_loss": avg_vloss,
                    "train_loss": avg_tloss,
                    "task_validation_loss": val_dict,
                    "task_train_loss": train_dict,
                    "learning_rate": current_lr[0],
                }

        loss_data.append(epoch_loss_dict)

        if training_config.wandb_active:
            wandb.log( epoch_loss_dict)


# undo the normalization ??
final_loss = avg_vloss
print(final_loss)


#torch.save(model.state_dict(), f"{training_config.training_data_dir}/regression_model.pth")
#
#torch.save(
#    model.preprocessor.state_dict(), f"{training_config.training_data_dir}/preprocessor.pth"
#)
#torch.save(model.encoder.state_dict(), f"{training_config.training_data_dir}/encoder.pth")
#
pyaml.to_yaml_file(f"{training_config.training_data_dir}/training_config.yaml", training_config)


architecture_config.reload_full_model_weights = f"{training_config.training_data_dir}/regression_model.pth"

pyaml.to_yaml_file(f"{training_config.training_data_dir}/architecture_config.yaml", architecture_config)



pyaml.to_yaml_file(f"{training_config.training_data_dir}/train_dataset_config.yaml", train_dataset.dataset_config)


pyaml.to_yaml_file(f"{training_config.training_data_dir}/valid_dataset_config.yaml", valid_dataset.dataset_config)


with open(f"{training_config.training_data_dir}/training_losses.yaml","x") as f:
    yaml.safe_dump(loss_data, f)



training_evaluation_pipeline = regression_pipeline(train_dataset, valid_dataset)
training_evaluation_pipeline.evaluate(model)
figs = training_evaluation_pipeline.visualize(output_directory=training_config.training_data_dir, model_name= run_name)

#if training_config.wandb_active:
#    for figname, figure in figs.items():
#        wandb.log({figname: figure})
