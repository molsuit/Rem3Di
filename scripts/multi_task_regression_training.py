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
from threedscriptors.data_handling.dataset_builder import DatasetBuildingDirector
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.training.regression_training import multitask_masked_loss

training_config = TrainingConfig(
    batch_size=32,
    epochs=75,
    learning_rate=1e-4,
    wandb_active=False,
    mace_model_path="/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model",
    dataset_path="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/test",
    model_dir="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/test",
)


_, dataset = DatasetBuildingDirector.reload_dataset(
    directory=training_config.dataset_path, return_normalized_targets=True
)


training_data, validation_data = random_split(dataset, [0.7, 0.3])

training_config.total_steps = len(training_data) * training_config.epochs


training_loader = DataLoader(
    training_data, batch_size=training_config.batch_size, shuffle=True, drop_last=True
)
validation_loader = DataLoader(
    validation_data,
    batch_size=training_config.batch_size,
    shuffle=False,
    drop_last=False,
)

architecture_config = pyaml.parse_yaml_file_as(
    ArchitectureConfig,
    f"{training_config.model_dir}/architecture_config.yaml",
)

mb = ModelBuilder(architecture_config=architecture_config)
model = mb.build_model()

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


for epoch in range(training_config.epochs):
    running_tloss = 0.0
    weighed_loss_per_task_train = torch.zeros(
        len(dataset.dataset_config.tasks), device=device
    )
    model.train()
    optimizer.zero_grad()

    for _batch, (
        embeddings,
        padding_mask,
        regression_targets,
        regression_masks,
    ) in enumerate(training_loader):
        embeddings = embeddings.to(device)
        padding_mask = padding_mask.to(device)
        regression_targets = regression_targets.to(device)
        regression_masks = regression_masks.to(device)

        # TODO: Harmonize the definition of the padding mask. Torch True = padded, prev: True = not padded

        prediction = model(embeddings, padding_mask=padding_mask)

        loss, batch_weighed_loss_per_task_train = multitask_masked_loss(
            predictions=prediction,
            labels=regression_targets,
            regression_mask=regression_masks,
        )

        weighed_loss_per_task_train += batch_weighed_loss_per_task_train.detach()

        loss.backward()
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
        for _batch, (
            embeddings,
            padding_mask,
            regression_targets,
            regression_masks,
        ) in enumerate(validation_loader):
            embeddings = embeddings.to(device)
            padding_mask = padding_mask.to(device)
            regression_targets = regression_targets.to(device)
            regression_masks = regression_masks.to(device)

            prediction = model(embeddings, padding_mask=padding_mask)

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
                    dataset.dataset_config.target_cols,
                    weighed_loss_per_task_val.cpu().detach().numpy()
                    * dataset.dataset_config.std,
                    strict=False,
                )
            )

            train_dict = dict(
                zip(
                    dataset.dataset_config.target_cols,
                    weighed_loss_per_task_train.cpu().detach().numpy()
                    * dataset.dataset_config.std,
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
