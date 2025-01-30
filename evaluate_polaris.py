import polaris as po 
import numpy as np

from data_handling.Dataset import DatasetFactory

import torch
from model.model import TransformerEncoder
from model.regression_heads import SingleRegressionModel
import json 

from analysis import plot_delta_histogram

# Load the benchmark from the Hub
benchmark = po.load_benchmark("biogen/adme-fang-SOLU-reg-v1")

# Get the train and test data-loaders
train , test = benchmark.get_train_test_split()

MACE_PATH = "/home/steffen/projects/mol_descriptors/mace_model/2023-12-10-mace-128-L0_energy_epoch-249.model"



#smiles_iterator = ListSmilesIterator(smiles)
dataset = DatasetFactory.from_disk("/home/steffen/projects/mol_descriptors/data/adme-fang-v1-solubility")
dataset.normalize_targets()

train_data_metadata = dataset.metadata

architecture_parameter = json.load(open("/home/steffen/projects/mol_descriptors/transformer_model/adme-fang-sol/architecture_parameters.json","r"))

model_parameters = torch.load("/home/steffen/projects/mol_descriptors/transformer_model/adme-fang-sol/regression_model.pth")

encoder = TransformerEncoder(num_layers=2, **architecture_parameter)
model = SingleRegressionModel(hidden_dim=256, output_dim=1, encoder=encoder)
model.load_state_dict(model_parameters)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.float().to(device)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

predictions_stack = None

for batch, (embeddings, padding_masks, _ , _) in enumerate(dataloader):
    embeddings = embeddings.float().to(device)
    padding_masks = padding_masks.float().to(device)

    predictions = model(embeddings, padding_masks)

    if predictions_stack is None:
        predictions_stack = predictions
    else:
        predictions_stack = torch.cat([predictions_stack, predictions], dim=0)


predictions = predictions_stack.cpu().detach().numpy()

predictions = (predictions * train_data_metadata["std"].numpy()) + train_data_metadata["mean"].numpy()



delta = predictions - train.targets[:750].reshape(-1,1)

histogram = plot_delta_histogram(delta)
histogram.savefig("./figures/delta_histogram.png")

mae = np.mean(np.abs(delta))
mse = np.mean((delta)**2)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
