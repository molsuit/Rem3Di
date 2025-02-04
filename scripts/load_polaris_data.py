import polaris as po
from mace.calculators import MACECalculator

from threedprints.data_handling.dataset import DatasetFactory
from threedprints.data_handling.data_config import DatasetConfig

from threedprints.data_handling.polaris_helper import pretreat_polaris_dataset
from threedprints.data_handling.preprocessing import (
    get_max_molecule_size,
)
from threedprints.data_handling.smiles_iterator import ListSmilesIterator
from dataclasses import asdict


# Load the benchmark from the Hub
dataset = po.load_dataset("biogen/adme-fang-v1")
dataset.cache()

smiles = dataset.table['MOL_smiles'].to_list()
target_cols = ["LOG_RLM_CLint", "LOG_SOLUBILITY"]
targets = dataset.table[target_cols].to_numpy()

#benchmark = po.load_benchmark("biogen/adme-fang-SOLU-reg-v1")
## Get the train and test data-loaders
#train, test = benchmark.get_train_test_split()
#smiles = train.inputs
##targets = train.targets
#
#print(smiles)
#print(targets)
#print(train.target_cols)


MODEL_DIR = "/data/fast-pc-06/snw30/projects/models"
MACE_PATH = f"{MODEL_DIR}/mace-omat-0-medium.model"

mace_calculator = MACECalculator(model_path=MACE_PATH, device="cuda",enable_cueq=True)

dataset_config = DatasetConfig(target_cols,N_molecules=1000,embedding_size=256,max_atoms=None,BFGS_max_steps=250,BFGS_tol=0.05,dataset_type="Regression")

if dataset_config.max_atoms is None:
    smiles_iterator = ListSmilesIterator(smiles)
    dataset_config.max_atoms = get_max_molecule_size(smiles_iterator)

metadata = asdict(dataset_config)

smiles, regression_targets, regression_masks, metadata = pretreat_polaris_dataset(
    smiles, targets, metadata
)


smiles_iterator = ListSmilesIterator(smiles)
dataset = DatasetFactory.from_smiles(
    smiles_iterator, mace_calculator, metadata, regression_targets, regression_masks
)

dataset.store_data_to_disk(
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/adme-fang-v1"
)
