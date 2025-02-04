from dataclasses import asdict

import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, random_split

from threedprints.data_handling.dataset import DatasetFactory
from threedprints.model.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
)
from threedprints.model.model import Transformer, TransformerDecoder, TransformerEncoder
from threedprints.training.pretraining import (
    train_loop,
    validation_loop,
)
from threedprints.training.training_config import TrainingConfig
from threedprints.utils.config_utils import to_yaml

MODEL_DIR = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/adme-fang-sol"
DATA_DIR = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data"

# TODO: Recommend making this into a dataclass, defined next to the model


attention_layer_config = AttentionLayerConfig(
    input_dim=256, num_heads=8, dim_feedforward=64, embedding_dim=256
)
architecture_config = ArchitectureConfig(
    N_layers=2, attention_layer=attention_layer_config
)
training_config = TrainingConfig(
    batch_size=16, masking_probability=0.15, epochs=100, learning_rate=1e-3
)

dataset = DatasetFactory.from_disk(DATA_DIR)

training_data, validation_data, test_data = random_split(dataset, [0.8, 0.1, 0.1])

training_config.total_steps = len(training_data) * training_config.epochs

training_loader = DataLoader(
    training_data, batch_size=training_config.batch_size, shuffle=True, drop_last=True
)
validation_loader = DataLoader(
    training_data, batch_size=training_config.batch_size, shuffle=True, drop_last=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = TransformerEncoder(
    num_layers=architecture_config.N_layers, **asdict(attention_layer_config)
)
decoder = TransformerDecoder(
    num_layers=architecture_config.N_layers, **asdict(attention_layer_config)
)
model = Transformer(encoder=encoder, decoder=decoder).to(device)


num_opt_steps = training_config.epochs * len(training_loader)
optimizer = optim.AdamW(model.parameters(), lr=training_config.learning_rate)
scheduler = OneCycleLR(
    optimizer, max_lr=training_config.learning_rate, total_steps=num_opt_steps
)

for epoch in range(training_config.epochs):
    train_loss = train_loop(
        training_loader, model, optimizer, scheduler, training_config, device
    )
    validation_loss = validation_loop(validation_loader, model, training_config, device)

    print(
        f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Validation Loss: {validation_loss:.3f}"
    )


torch.save(
    model.state_dict(), f"{MODEL_DIR}/transformer_model.pth"
)  # Store the whole model

torch.save(
    model.encoder.state_dict(), f"{MODEL_DIR}/transformer_encoder.pth"
)  # Store the encoder only

to_yaml(f"{MODEL_DIR}/architecture_config.yaml", architecture_config)
