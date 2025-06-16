from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset import RegressionDataset
from threedscriptors.data_handling.dataset_io import store_data_to_disk, load_data_from_disk
from threedscriptors.data_handling.pipelines import regression_training_pipeline
from threedscriptors.data_handling.source_preprocessing.polaris_preprocessing import (
    load_polaris_dataset,
)

from dataclasses import asdict


dataset_registry = {
    "antiviral_admet": "asap-discovery/antiviral-admet-2025-unblinded",
    "adme_fang": "biogen/adme-fang-v1",
}

smiles_column = {
    "antiviral_admet": "CXSMILES",
    "adme_fang": "MOL_smiles",
}

non_task_columns = {
    "antiviral_admet": ["Molecule Name","Set", "CXSMILES"],
    "adme_fang": ["UNIQUE_ID", "MOL_smiles", "SMILES"],
}


load_dataset = "antiviral_admet"

# Load the benchmark from polarishub
smiles, regression_targets, regression_masks, tasks = load_polaris_dataset(
    dataset_registry[load_dataset],
    smiles_column=smiles_column[load_dataset],
    non_task_columns=non_task_columns[load_dataset],datasplit="train"
)
print(len(smiles))

dataset_directory = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{load_dataset}_test"

MACE_PATH = (
    "/share/snw30/projects/mace_model/mace_agnesi_medium.model"
)
embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_mp_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)

dataset_config = DatasetConfig(
    N_molecules=200,
    dataset_type=RegressionDataset,
    BFGS_tol=0.2,
    BFGS_max_steps=500,
    N_conformers=2,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
)


dataset = regression_training_pipeline(
    dataset_config, smiles, regression_targets, regression_masks
).build()

store_data_to_disk(dataset, f"{dataset_directory}_full")


print(dataset.embeddings.shape)
print(dataset.regression_masks.shape)
print(dataset.regression_targets.shape)


from threedscriptors.data_handling.dataset_io import store_data_to_disk, load_data_from_disk

dataset_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/adme_fang"
dataset = load_data_from_disk(f"{dataset_directory}_full")


splitting_ratio = [0.8,0.2]

split_datasets = dataset.split_dataset(splitting_ratio)

store_data_to_disk(split_datasets[0], f"{dataset_directory}_train")
store_data_to_disk(split_datasets[1], f"{dataset_directory}_valid")


