from dataclasses import asdict

import torch
from mace.calculators import mace_off
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

import wandb
from threedscriptors.data_handling.dataset import DatasetFactory
from threedscriptors.model.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
)
from threedscriptors.model.model import TransformerEncoder
from threedscriptors.model.regression_heads import (
    MultiTaskRegressionModel,
)
from threedscriptors.training.regression_training import multitask_masked_loss
from threedscriptors.training.training_config import TrainingConfig
from threedscriptors.utils.config_utils import get_global_config, to_yaml

MODEL_DIR = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/adme-fang-sol"

training_config = TrainingConfig(
    batch_size=32, epochs=250, learning_rate=1e-4, wandb_active=True
)

attention_layer_config = AttentionLayerConfig(
    input_dim=128, num_heads=8, dim_feedforward=256, embedding_dim=128, dropout=0.3
)

architecture_config = ArchitectureConfig(
    N_layers=2, attention_layer=attention_layer_config
)


# TODO: recommend to define scripts in a main function, then do the following:
# if __name__ == "__main__":
#     main()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
mace_calculator = mace_off("medium", device, enable_cueq=True)

dataset = DatasetFactory.from_disk(
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/adme-fang-v1"
)
dataset.dataset_config.embedding_size = 128
dataset.normalize_targets()
dataset.calculate_embeddings(mace_calculator)

config_dict = get_global_config(
    training_config, dataset.dataset_config, architecture_config
)

if training_config.wandb_active:
    wandb.init(project="threedscriptors", config=config_dict)


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

encoder = TransformerEncoder(
    num_layers=architecture_config.N_layers,
    **asdict(attention_layer_config),
)


model = MultiTaskRegressionModel(
    hidden_dim=256,
    output_dim=1,
    encoder=encoder,
    task_list=dataset.dataset_config.target_cols,
)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))


num_opt_steps = training_config.epochs * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
scheduler = OneCycleLR(
    optimizer, max_lr=training_config.learning_rate, total_steps=num_opt_steps
)

model.to(device)


for epoch in range(training_config.epochs):
    running_tloss = 0.0
    weighed_loss_per_task_train = torch.zeros(
        len(dataset.dataset_config.target_cols), device=device
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

        weighed_loss_per_task_train += batch_weighed_loss_per_task_train

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_tloss += loss.item()

    avg_tloss = (
        running_tloss / (_batch + 1)
    )  # TODO: if you don't have anything else to do, you could use TorchMetrics to calculate the loss. It's a bit of a hassle to set up though

    running_vloss = 0.0
    weighed_loss_per_task_val = torch.zeros(
        len(dataset.dataset_config.target_cols), device=device
    )

    model.eval()

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

    print(f"Epoch {epoch} Training Loss: {avg_tloss} Validation Loss: {avg_vloss}")

    if training_config.wandb_active:
        val_dict = dict(
            zip(
                dataset.dataset_config.target_cols,
                weighed_loss_per_task_val.cpu().detach().numpy(),
                strict=False,
            )
        )

        train_dict = dict(
            zip(
                dataset.dataset_config.target_cols,
                weighed_loss_per_task_train.cpu().detach().numpy(),
                strict=False,
            )
        )

        wandb.log(
            {
                "validation_loss": avg_vloss,
                "train_loss": avg_tloss,
                "task_validation_loss": val_dict,
                "task_train_loss": train_dict,
            }
        )


# undo the normalization ??
final_loss = avg_vloss
print(final_loss)


torch.save(model.state_dict(), f"{MODEL_DIR}/regression_model.pth")

to_yaml(f"{MODEL_DIR}/architecture_config.yaml", architecture_config)
