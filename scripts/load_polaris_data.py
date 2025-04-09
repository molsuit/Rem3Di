from threedscriptors.configuration.data_config import DatasetConfig, DatasetTypes
from threedscriptors.data_handling.dataset_factory import DatasetBuildingDirector
from threedscriptors.data_handling.polaris_preprocessing import load_polaris_dataset
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator

# Load the benchmark from polarishub
dataset, tasks = load_polaris_dataset("asap-discovery/antiviral-admet-2025-unblinded")

dataset_directory = (
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/test"
)

MACE_PATH = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)
# Get the train and test data-loaders

dataset_config = DatasetConfig(
    N_molecules=50,
    dataset_type=DatasetTypes.REGRESSION,
    BFGS_tol=0.2,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model=MACE_PATH,
    max_atoms=None,
    tasks=tasks,
)

smiles_iterator = ListSmilesIterator(dataset.smiles)
_, dataset = DatasetBuildingDirector.build_dataset(
    iterator=smiles_iterator,
    dataset_config=dataset_config,
    regression_targets=dataset.regression_targets,
    regression_masks=dataset.regression_masks,
)

dataset.store_data_to_disk(dataset_directory)
