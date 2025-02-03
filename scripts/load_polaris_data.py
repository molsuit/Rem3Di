import polaris as po
from mace.calculators import MACECalculator

from threedprints.data_handling.dataset import DatasetFactory
from threedprints.data_handling.polaris_helper import pretreat_polaris_dataset
from threedprints.data_handling.preprocessing import (
    get_max_molecule_size,
)
from threedprints.data_handling.smiles_iterator import ListSmilesIterator

# Load the benchmark from the Hub
benchmark = po.load_benchmark("biogen/adme-fang-SOLU-reg-v1")

# Get the train and test data-loaders
train, test = benchmark.get_train_test_split()

smiles = train.inputs
targets = train.targets

print(smiles)
print(targets)
print(train.target_cols)

MACE_PATH = "/home/steffen/projects/mol_descriptors/mace_model/2023-12-10-mace-128-L0_energy_epoch-249.model"

mace_calculator = MACECalculator(model_path=MACE_PATH, device="cuda", enable_cueq=True)

metadata = {
    "max_atoms": 93,
    "BFGS_tol": 0.05,
    "BFGS_max_steps": 250,
    "dataset_type": "Regression",
    "N_molecules": 750,
    "embedding_size": 256,
    "target_cols": ["LOG_SOLUBILITY"],
}

smiles_iterator = ListSmilesIterator(smiles)
max_size = get_max_molecule_size(smiles_iterator)
print(max_size)

smiles, regression_targets, regression_masks, metadata = pretreat_polaris_dataset(
    smiles, targets, metadata
)


smiles_iterator = ListSmilesIterator(smiles)
dataset = DatasetFactory.from_smiles(
    smiles_iterator, mace_calculator, metadata, regression_targets, regression_masks
)

dataset.store_data_to_disk(
    "/home/steffen/projects/mol_descriptors/data/adme-fang-v1-solubility"
)
