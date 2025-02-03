import json

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

from threedprints.data_handling.dataset import DatasetFactory
from threedprints.model.model import TransformerEncoder
from threedprints.model.regression_heads import SingleRegressionModel

MODEL_DIR = "/home/steffen/projects/mol_descriptors/transformer_model/adme-fang-sol"

hyperparameter = {
    "batch_size": 64,
    "masking_probability": 0.15,
    "epochs": 1000,
    "learning_rate": 1e-4,
}

# TODO: Recommend making this into a dataclass, defined next to the model
architecture_parameter = {
    "input_dim": 256,
    "num_heads": 8,
    "dim_feedforward": 128,
    "embedding_dim": 256,
}

# TODO: recommend to define scripts in a main function, then do the following:
# if __name__ == "__main__":
#     main()

dataset = DatasetFactory.from_disk(
    "/home/steffen/projects/mol_descriptors/data/adme-fang-v1-solubility"
)

dataset.normalize_targets()

training_data, validation_data = random_split(dataset, [0.7, 0.3])

hyperparameter["total_steps"] = len(training_data) * hyperparameter["epochs"]

training_loader = DataLoader(
    training_data, batch_size=hyperparameter["batch_size"], shuffle=True, drop_last=True
)
validation_loader = DataLoader(
    training_data, batch_size=hyperparameter["batch_size"], shuffle=True, drop_last=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = TransformerEncoder(num_layers=2, **architecture_parameter)


model = SingleRegressionModel(hidden_dim=256, output_dim=1, encoder=encoder).to(device)


num_opt_steps = hyperparameter["epochs"] * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=hyperparameter["learning_rate"])
scheduler = OneCycleLR(
    optimizer, max_lr=hyperparameter["learning_rate"], total_steps=num_opt_steps
)


loss_fn = nn.MSELoss()

for epoch in range(hyperparameter["epochs"]):
    running_tloss = 0.0

    model.train()
    optimizer.zero_grad()

    for batch, (
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

        prediction = model(embeddings, padding_mask=torch.logical_not(padding_mask))

        loss = loss_fn(prediction, regression_targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_tloss += loss.item()

    avg_tloss = (
        running_tloss / (batch + 1)
    )  # TODO: if you don't have anything else to do, you could use TorchMetrics to calculate the loss. It's a bit of a hassle to set up though

    running_vloss = 0.0
    model.eval()

    for batch, (
        embeddings,
        padding_mask,
        regression_targets,
        regression_masks,
    ) in enumerate(validation_loader):
        embeddings = embeddings.to(device)
        padding_mask = padding_mask.to(device)
        regression_targets = regression_targets.to(device)
        regression_masks = regression_masks.to(device)

        prediction = model(embeddings, padding_mask=torch.logical_not(padding_mask))

        loss = loss_fn(prediction, regression_targets)

        running_vloss += loss.item()

    avg_vloss = running_vloss / (batch + 1)
    print(f"Epoch {epoch} Training Loss: {avg_tloss} Validation Loss: {avg_vloss}")

# undo the normalization
final_loss = avg_vloss * dataset.metadata["std"][0]
print(final_loss)


torch.save(model.state_dict(), f"{MODEL_DIR}/regression_model.pth")

with open(f"{MODEL_DIR}/architecture_parameters.json", "w") as f:
    json.dump(architecture_parameter, f)  # Store architecture parameters
