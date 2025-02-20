from dataclasses import asdict

import torch
from mace.calculators import MACECalculator
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

import wandb
from threedscriptors.data_handling.dataset import DatasetFactory
from threedscriptors.model.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    GlobalAggregatorConfig,
    RegressionHeadConfig,
)
from threedscriptors.model.atomic_descriptor_preprocess import (
    InvariantsFilter,
    PseudoscalarGenerator,
)
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.regression_models import (
    MultiTaskRegressionModel,
)
from threedscriptors.model.transformer_components import TransformerEncoder
from threedscriptors.training.regression_training import multitask_masked_loss
from threedscriptors.training.training_config import TrainingConfig
from threedscriptors.utils.config_utils import get_global_config, to_yaml
from threedscriptors.utils.model_utils import get_mace_calculator_irrep_signature

MODEL_DIR = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/antiviral-admet"

training_config = TrainingConfig(
    batch_size=32,
    epochs=150,
    learning_rate=1e-4,
    wandb_active=True,
    mace_model_path="/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model",
    dataset_path="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/antiviral-admet",
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
mace_calculator = MACECalculator(
    model_paths=training_config.mace_model_path, device=device, enable_cueq=True
)

calculator_irreps = get_mace_calculator_irrep_signature(mace_calculator)


embedding_preprocessor_config = EmbeddingPreprocessConfig(
    input_irreps=calculator_irreps, pseudoscalars=True
)


# TODO: recommend to define scripts in a main function, then do the following:
# if __name__ == "__main__":
#     main()


dataset = DatasetFactory.from_disk(directory=training_config.dataset_path)

dataset.normalize_targets()
dataset.calculate_embeddings(mace_calculator, embedding_size=calculator_irreps.dim)


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

# Refactor all Config into a ModelFactory?
if embedding_preprocessor_config.pseudoscalars:
    preprocessor = PseudoscalarGenerator(embedding_preprocessor_config)
else:
    preprocessor = InvariantsFilter(embedding_preprocessor_config)


attention_layer_config = AttentionLayerConfig(
    input_dim=preprocessor.config.output_irreps_dim,
    num_heads=8,
    dim_feedforward=512,
    embedding_dim=preprocessor.config.output_irreps_dim,
    dropout=0.3,
)

architecture_config = ArchitectureConfig(
    N_layers=2, attention_layer=attention_layer_config
)

regression_head_config = RegressionHeadConfig(
    activation_fn=nn.SiLU(), hidden_dimensions=[512, 256, 128]
)

global_aggregator_config = GlobalAggregatorConfig(
    input_dim=attention_layer_config.embedding_dim,
    aggregation_fn=[torch.mean],
)


global_aggregator = GlobalAggregator(global_aggregator_config)

encoder = TransformerEncoder(
    architecture_config=architecture_config,
    **asdict(attention_layer_config),
)

model = MultiTaskRegressionModel(
    regression_head_config=regression_head_config,
    encoder=encoder,
    task_list=dataset.dataset_config.target_cols,
    preprocessor=preprocessor,
    global_aggregator=global_aggregator,
)


config_dict = get_global_config(
    training_config, dataset.dataset_config, architecture_config, regression_head_config
)

if training_config.wandb_active:
    wandb.init(project="threedscriptors", entity="threedscriptors", config=config_dict)


model.to(device)

num_opt_steps = training_config.epochs * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
scheduler = OneCycleLR(
    optimizer, max_lr=training_config.learning_rate, total_steps=num_opt_steps
)


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
        len(dataset.dataset_config.target_cols), device=device
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
                    * dataset.dataset_config.std.cpu().detach().numpy(),
                    strict=False,
                )
            )

            train_dict = dict(
                zip(
                    dataset.dataset_config.target_cols,
                    weighed_loss_per_task_train.cpu().detach().numpy()
                    * dataset.dataset_config.std.cpu().detach().numpy(),
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


torch.save(model.state_dict(), f"{MODEL_DIR}/regression_model.pth")

to_yaml(f"{MODEL_DIR}/architecture_config.yaml", architecture_config)
