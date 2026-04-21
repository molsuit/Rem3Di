from pathlib import Path

from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset import (
    RegressionDatasetwithRandomWalks,
)
from threedscriptors.data_handling.dataset_builder import DatasetBuilder
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import (
    regression_training_from_structures_pipeline,
)
from threedscriptors.data_handling.source_preprocessing.qm9_preprocessing import (
    QM9PropertyNames,
    load_qm9,
)

qm9_dir = Path("/home/snw30/rds/hpc-work/3DMolecularDescriptors/data/raw_data/qm9_raw")


N_molecules = 134000

tasks_to_load = [
    QM9PropertyNames.mu,
    QM9PropertyNames.gap,
    QM9PropertyNames.alpha,
    QM9PropertyNames.r2,
    QM9PropertyNames.zpve,
    QM9PropertyNames.Cv,
]

smiles, molecules, structure_ids, regression_targets, regression_masks, task_configs = (
    load_qm9(qm9_dir, N_molecules, tasks_to_load=tasks_to_load)
)

assert regression_targets.shape[1] == len(tasks_to_load)

dataset_directory = "/home/snw30/rds/hpc-work/3DMolecularDescriptors/data/qm9_full"

MACE_PATH = "/home/snw30/rds/hpc-work/models/MACE-OFF24_medium.model"

embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_off_24_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)

dataset_config = DatasetConfig(
    N_molecules=N_molecules,
    dataset_type=RegressionDatasetwithRandomWalks,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=task_configs,
    only_heavy_atoms=False,
    dataset_name="qm9",
    rw_transition_matrix_from_3D=True,
)


dataset = regression_training_from_structures_pipeline(
    dataset_config, molecules, structure_ids, regression_targets, regression_masks
).build()


store_data_to_disk(dataset, dataset_directory + "_full")


from threedscriptors.training.dataset_splitting import DatasetSplitting

ds = DatasetSplitting(dataset)
names = ["training", "test"]
split_ratios = [0.9, 0.1]
split_dataset_indices = ds.general_split(split_ratios, True)


for ids, name in zip(split_dataset_indices, names, strict=False):
    new_dataset = ds.materialise_dataset_split(dataset, ids)

    db = DatasetBuilder(new_dataset)
    db.canonicalize_structure_ids()
    store_data_to_disk(new_dataset, dataset_directory + "_" + name)
