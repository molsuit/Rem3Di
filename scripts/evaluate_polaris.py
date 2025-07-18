from dataclasses import asdict

import numpy as np
import polaris as po
import torch
import yaml
from mace.calculators import MACECalculator
from torch import nn
from torch.utils.data import DataLoader

from threedscriptors.configuration.training_config import TrainingConfig
from threedscriptors.data_handling.data_config import DatasetConfig
from threedscriptors.data_handling.dataset import DatasetFactory
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.model.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    GlobalAggregatorConfig,
    RegressionHeadConfig,
)
from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    InvariantsFilter,
    PseudoscalarGenerator,
)
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.regression_models import (
    MultiTaskRegressionModel,
)
from threedscriptors.model.encoder import TransformerEncoder
from threedscriptors.utils.model_utils import get_mace_calculator_irrep_signature

# Load the competition from the Hub
competition = po.load_competition("asap-discovery/antiviral-potency-2025")
# Get the train and test data-loaders
_, test = competition.get_train_test_split()

N_samples = test.as_dataframe().to_numpy().shape[0]


print(test.target_cols)


dataset_config = DatasetConfig(
    N_molecules=None,
    max_atoms=138,
    BFGS_max_steps=2000,
    BFGS_tol=0.5,
    dataset_type="Evaluation",
    N_conformers=10,
    target_cols=test.target_cols,
)

dataset_config.N_molecules = N_samples * dataset_config.N_conformers

data = test.as_dataframe()
smiles = data["CXSMILES"]


training_config = TrainingConfig(
    batch_size=32,
    epochs=125,
    learning_rate=1e-4,
    wandb_active=True,
    mace_model_path="/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model",
    dataset_path="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/antiviral-potency/test",
    model_dir="/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/antiviral-potency",
)

train_config = yaml.safe_load(open(f"{training_config.model_dir}/config.yaml"))

mean = np.array(train_config["dataset_config"]["mean"]).reshape(1, -1)
std = np.array(train_config["dataset_config"]["std"]).reshape(1, -1)
max_atoms = train_config["dataset_config"]["max_atoms"]

train_tasks = train_config["dataset_config"]["target_cols"]


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
mace_calculator = MACECalculator(
    model_paths=training_config.mace_model_path, device=device, enable_cueq=True
)

calculator_irreps = get_mace_calculator_irrep_signature(mace_calculator)


embedding_preprocessor_config = EmbeddingPreprocessConfig(
    input_irreps=calculator_irreps, pseudoscalars=True
)

iterator = ListSmilesIterator(smiles)
# testdata = DatasetFactory.from_smiles(iterator,mace_calculator,dataset_config)
# testdata.store_data_to_disk(f"{training_config.dataset_path}/")
testdata = DatasetFactory.from_disk(f"{training_config.dataset_path}/")
print(len(testdata.smiles_list))

testdata.calculate_embeddings(mace_calculator, embedding_size=calculator_irreps.dim)


eval_loader = DataLoader(testdata, batch_size=32, shuffle=False, drop_last=False)


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
    task_list=train_tasks,
    preprocessor=preprocessor,
    global_aggregator=global_aggregator,
)


state_dict = torch.load(f"{training_config.model_dir}/regression_model.pth")
model.load_state_dict(state_dict)


model.to(device)
predictions_stack = None
model.eval()

for _, (embeddings, padding_masks) in enumerate(eval_loader):
    with torch.no_grad():
        embeddings = embeddings.to(device)
        padding_masks = padding_masks.to(device)

        predictions = model(embeddings, padding_masks)

        if predictions_stack is None:
            predictions_stack = predictions
        else:
            predictions_stack = torch.cat([predictions_stack, predictions], dim=0)


predictions = predictions_stack.cpu().detach().numpy()
predictions = (predictions * std) + mean


predictions_chunked = predictions.reshape(
    -1, dataset_config.N_conformers, predictions.shape[1]
)
chunks_mean = predictions_chunked.mean(axis=1).squeeze()
chunks_std = predictions_chunked.std(axis=1).squeeze()

mean_std_ratio_per_task = [
    np.mean(chunks_std[:, i] / std[:, i]) for i in range(std.shape[1])
]
max_std_deviation_ratio_per_task = [
    np.max(chunks_std[:, i] / std[:, i]) for i in range(std.shape[1])
]  # The ratio between std deviation of the chunks (predictions of all conformers and the total train dataset)
print("Max Std per task")
print(max_std_deviation_ratio_per_task)
print("Mean Std per task")
print(mean_std_ratio_per_task)

predictions = chunks_mean
# admet 93, potency 99
predictions = np.insert(predictions, 99, mean, axis=0)
print(np.mean(predictions, axis=0))

predictions = {col: predictions[:, i] for i, col in enumerate(train_tasks)}


for key, value in predictions.items():
    print(key)
    print(np.mean(value))

print(predictions)
