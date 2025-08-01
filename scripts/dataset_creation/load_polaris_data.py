
from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    MaceCalculatorConfig,
    DatasetSplit
)
from threedscriptors.data_handling.dataset import (
    RegressionDatasetwithPositions,RegressionDatasetwithRandomWalks
)
from threedscriptors.data_handling.dataset_analysis import DatasetPostLoadAnalysis

from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.data_handling.dataset_io import (
    store_data_to_disk,
)
from threedscriptors.data_handling.pipelines import (
    regression_training_with_pos_pipeline
)
from threedscriptors.data_handling.source_preprocessing.polaris_preprocessing import (
    load_polaris_dataset,
)

dataset_registry = {
    "antiviral_admet": "asap-discovery/antiviral-admet-2025-unblinded",
    "adme_fang": "biogen/adme-fang-v1",
    "antiviral_potency" : "asap-discovery/antiviral-potency-2025-unblinded"
}

smiles_column = {
    "antiviral_admet": "CXSMILES",
    "adme_fang": "MOL_smiles",
    "antiviral_potency": "CXSMILES"
}

non_task_columns = {
    "antiviral_admet": ["Molecule Name","Set", "CXSMILES"],
    "adme_fang": ["UNIQUE_ID", "MOL_smiles", "SMILES"],
    "antiviral_potency": ["Molecule Name","Set", "CXSMILES"],

}


load_dataset = "antiviral_admet"
# Load the benchmark from polarishub
smiles, regression_targets, regression_masks, tasks = load_polaris_dataset(
    dataset_registry[load_dataset],
    smiles_column=smiles_column[load_dataset],
    non_task_columns=non_task_columns[load_dataset],datasplit="Test"
)


dataset_directory = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{load_dataset}_test"

MACE_PATH = (
    "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
)

embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_off_24_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda"
)

dataset_config = DatasetConfig(
    N_molecules=5000,
    dataset_type=RegressionDatasetwithPositions,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=10,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
    only_heavy_atoms=False,
    dataset_name= load_dataset,
)


pipeline = regression_training_with_pos_pipeline(
    dataset_config, smiles, regression_targets, regression_masks
) 

dataset = pipeline.build()

store_data_to_disk(dataset, f"{dataset_directory}_full")

DatasetPostLoadAnalysis(dataset, f"{dataset_directory}_full").run()
