import numpy as np
import pydantic_yaml as pyaml
import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

import wandb
from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.dataset import (
    RegressionWithAuxDataset,
)
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.data_handling.sample import sample_collate_fn
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.regression_training import multitask_masked_loss

torch.manual_seed(0)

np.random.seed(0)


training_config = TrainingConfig(
    batch_size=128,
    epochs=100,
    learning_rate=2e-5,
    max_grad_norm=1.0,
    wandb_active=True,
    mace_model_path="/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model",
    dataset_path="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt",
    model_dir="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_ps",
    normalized_atomic_descriptors=True,
    normalized_targets=True,
)


pipeline_orchestrator = reload_dataset_pipeline(
    training_config.dataset_path,
    normalize_inputs=training_config.normalized_atomic_descriptors,
    normalize_targets=training_config.normalized_targets,
    dataset_cls=RegressionWithAuxDataset,
)
dataset = pipeline_orchestrator.build()
mean_embeddings, std_embeddings = (
    pipeline_orchestrator.builder.get_mean_and_std_embeddings()
)

# dataset = reload_dataset_pipeline(training_config.dataset_path, normalize_inputs= training_config.normalized_atomic_descriptors, normalize_targets= training_config.normalized_targets, dataset_cls= RegressionDataset).build()


training_data, validation_data = random_split(dataset, [0.8, 0.2])

training_config.total_steps = len(training_data) * training_config.epochs


training_loader = DataLoader(
    training_data,
    batch_size=training_config.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    collate_fn=sample_collate_fn,
)
validation_loader = DataLoader(
    validation_data,
    batch_size=training_config.batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=sample_collate_fn,
)

architecture_config = pyaml.parse_yaml_file_as(
    ArchitectureConfig,
    f"{training_config.model_dir}/architecture_config.yaml",
)

mb = ModelBuilder(architecture_config=architecture_config)
model = mb.build_model(
    mean_atomic_embedding=mean_embeddings, std_atomic_embedding=std_embeddings
)


print(f"Trainable Parameters: {mb.N_trainable_parameters}")


if training_config.wandb_active:
    wandb.init(
        project="threedscriptors",
        entity="threedscriptors",
        config={
            "architecture_config": architecture_config.model_dump(),
            "training_config": training_config.model_dump(),
            "dataset_config": dataset.dataset_config.model_dump(),
        },
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


num_opt_steps = training_config.epochs * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
scheduler = OneCycleLR(
    optimizer, max_lr=training_config.learning_rate, total_steps=num_opt_steps
)

task_names = [task.task_name for task in dataset.dataset_config.tasks]
stds = np.array([task.std for task in dataset.dataset_config.tasks])

print("Starting Training")
for epoch in range(training_config.epochs):
    running_tloss = 0.0
    weighed_loss_per_task_train = torch.zeros(
        len(dataset.dataset_config.tasks), device=device
    )
    model.train()
    optimizer.zero_grad()

    for _batch, samples in enumerate(training_loader):
        embeddings = samples.embeddings.to(device)
        padding_mask = samples.padding_mask.to(device)
        regression_targets = samples.regression_targets.to(device)
        regression_masks = samples.regression_masks.to(device)

        auxillary_data = samples.auxillary_data
        # TODO: Harmonize the definition of the padding mask. Torch True = padded, prev: True = not padded

        prediction = model(
            embeddings, padding_mask=padding_mask, auxillary_data=auxillary_data
        )

        loss, batch_weighed_loss_per_task_train = multitask_masked_loss(
            predictions=prediction,
            labels=regression_targets,
            regression_mask=regression_masks,
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
        len(dataset.dataset_config.tasks), device=device
    )

    model.eval()
    with torch.no_grad():
        for _batch, samples in enumerate(validation_loader):
            embeddings = samples.embeddings.to(device)
            padding_mask = samples.padding_mask.to(device)
            regression_targets = samples.regression_targets.to(device)
            regression_masks = samples.regression_masks.to(device)
            auxillary_data = samples.auxillary_data

            prediction = model(
                embeddings, padding_mask=padding_mask, auxillary_data=auxillary_data
            )

            loss, batch_weighed_loss_per_task_val = multitask_masked_loss(
                predictions=prediction,
                labels=regression_targets,
                regression_mask=regression_masks,
            )
            weighed_loss_per_task_val += batch_weighed_loss_per_task_val

            running_vloss += loss.item()

        avg_vloss = running_vloss / (_batch + 1)
        weighed_loss_per_task_val = weighed_loss_per_task_val / (_batch + 1)

        print(f"Epoch {epoch} Training Loss: {avg_tloss} Validation Loss: {avg_vloss}")

        current_lr = scheduler.get_last_lr()

        if training_config.wandb_active:
            val_dict = dict(
                zip(
                    task_names,
                    weighed_loss_per_task_val.cpu().detach().numpy() * stds,
                    strict=False,
                )
            )

            train_dict = dict(
                zip(
                    task_names,
                    weighed_loss_per_task_train.cpu().detach().numpy() * stds,
                    strict=False,
                )
            )

            wandb.log(
                {
                    "validation_loss": avg_vloss,
                    "train_loss": avg_tloss,
                    "task_validation_loss": val_dict,
                    "task_train_loss": train_dict,
                    "learning_rate": current_lr[0],
                }
            )


# undo the normalization ??
final_loss = avg_vloss
print(final_loss)


torch.save(model.state_dict(), f"{training_config.model_dir}/regression_model.pth")
torch.save(
    model.preprocessor.state_dict(), f"{training_config.model_dir}/preprocessor.pth"
)
torch.save(model.encoder.state_dict(), f"{training_config.model_dir}/encoder.pth")

pyaml.to_yaml_file(f"{training_config.model_dir}/training_config.yaml", training_config)
