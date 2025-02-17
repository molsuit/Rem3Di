from dataclasses import asdict

import torch
from mace.calculators import mace_off
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

from threedscriptors.data_handling.dataset import DatasetFactory
from threedscriptors.model.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
)
from threedscriptors.model.regression_models import SingleRegressionModel
from threedscriptors.model.transformer_components import TransformerEncoder
from threedscriptors.training.training_config import TrainingConfig
from threedscriptors.utils.config_utils import to_yaml

MODEL_DIR = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/adme-fang-sol"

training_config = TrainingConfig(batch_size=64, epochs=200, learning_rate=1e-3)

attention_layer_config = AttentionLayerConfig(
    input_dim=128, num_heads=8, dim_feedforward=64, embedding_dim=128, dropout=0.15
)
architecture_config = ArchitectureConfig(
    N_layers=1, attention_layer=attention_layer_config
)

# TODO: recommend to define scripts in a main function, then do the following:
# if __name__ == "__main__":
#     main()

dataset = DatasetFactory.from_disk(
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/adme-fang-v1"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

mace_calculator = mace_off("medium", device, enable_cueq=True)
dataset.normalize_targets()
dataset.calculate_embeddings(mace_calculator)

training_data, validation_data = random_split(dataset, [0.8, 0.2])

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

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = TransformerEncoder(
    num_layers=architecture_config.N_layers, **asdict(attention_layer_config)
)


model = SingleRegressionModel(hidden_dim=256, output_dim=1, encoder=encoder).to(device)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

num_opt_steps = training_config.epochs * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
scheduler = OneCycleLR(
    optimizer, max_lr=training_config.learning_rate, total_steps=num_opt_steps
)


loss_fn = nn.MSELoss()

for epoch in range(training_config.epochs):
    running_tloss = 0.0

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

        loss = loss_fn(prediction, regression_targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_tloss += loss.item()

    avg_tloss = (
        running_tloss / (_batch + 1)
    )  # TODO: if you don't have anything else to do, you could use TorchMetrics to calculate the loss. It's a bit of a hassle to set up though

    running_vloss = 0.0
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

        loss = loss_fn(prediction, regression_targets)

        running_vloss += loss.item()

    avg_vloss = running_vloss / (_batch + 1)
    print(f"Epoch {epoch} Training Loss: {avg_tloss} Validation Loss: {avg_vloss}")

# undo the normalization
final_loss = avg_vloss
print(final_loss)


torch.save(model.state_dict(), f"{MODEL_DIR}/regression_model.pth")

to_yaml(f"{MODEL_DIR}/architecture_config.yaml", architecture_config)
