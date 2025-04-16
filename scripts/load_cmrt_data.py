from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.cmrt_preprocessing import load_cmrt_data
from threedscriptors.data_handling.dataset_builder import (
    DatasetBuildingDirector,
)
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator

dataset_directory = (
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt"
)

smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data(
    f"{dataset_directory}/raw_data.csv", single_column_type=True
)

MACE_PATH = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)
# Get the train and test data-loaders

embedding_model_config = MaceCalculatorConfig(
    model_path=MACE_PATH, enable_cueq=True, device="cuda"
)


dataset_config = DatasetConfig(
    N_molecules=1000,
    dataset_type=DatasetTypes.REGRESSION,
    BFGS_tol=0.2,
    BFGS_max_steps=500,
    N_conformers=32,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
)

iterator = ListSmilesIterator(smiles)
_, dataset = DatasetBuildingDirector.build_chiral_dataset(
    iterator=iterator,
    dataset_config=dataset_config,
    regression_targets=regression_targets,
    regression_masks=regression_masks,
    auxillary_data=aux_data,
)
dataset.store_data_to_disk(dataset_directory)
