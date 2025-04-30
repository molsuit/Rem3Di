from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import chiral_regression_training_pipeline
from threedscriptors.data_handling.source_preprocessing.cmrt_preprocessing import (
    load_cmrt_data,
)

dataset_directory = (
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt"
)

smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data(
    f"{dataset_directory}/raw_data.csv", single_column_type=True
)

MACE_PATH = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)

embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_mp_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)
dataset_config = DatasetConfig(
    N_molecules=2000,
    dataset_type=DatasetTypes.REGRESSION,
    BFGS_tol=0.2,
    BFGS_max_steps=500,
    N_conformers=16,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
)

dataset = chiral_regression_training_pipeline(
    dataset_config=dataset_config,
    smiles=smiles,
    regression_targets=regression_targets,
    regression_masks=regression_masks,
    auxillary_data=aux_data,
).build()

store_data_to_disk(dataset, dataset_directory)
