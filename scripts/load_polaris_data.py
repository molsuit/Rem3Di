from threedscriptors.configuration.data_config import DatasetConfig, DatasetTypes
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import regression_training_pipeline
from threedscriptors.data_handling.source_preprocessing.polaris_preprocessing import (
    load_polaris_dataset,
)

# Load the benchmark from polarishub
smiles, regression_targets, regression_masks, tasks = load_polaris_dataset(
    "asap-discovery/antiviral-admet-2025-unblinded"
)

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


dataset = regression_training_pipeline(
    dataset_config, smiles, regression_targets, regression_masks
).build()

store_data_to_disk(dataset, dataset_directory)
