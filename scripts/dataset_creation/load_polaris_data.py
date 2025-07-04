
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
    regression_training_with_pos_pipeline, regression_training_with_transition_probs_pipeline
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


load_dataset = "antiviral_potency"
# Load the benchmark from polarishub
smiles, regression_targets, regression_masks, tasks = load_polaris_dataset(
    dataset_registry[load_dataset],
    smiles_column=smiles_column[load_dataset],
    non_task_columns=non_task_columns[load_dataset],datasplit="Train"
)




dataset_directory = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{load_dataset}"

MACE_PATH = (
    "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
)
embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_off24_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)

dataset_config = DatasetConfig(
    N_molecules=1031,
    dataset_type=RegressionDatasetwithPositions,#RegressionDatasetwithRandomWalks,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
    only_heavy_atoms=False,
    load_adjacency_matrix=True, 
    dataset_name= load_dataset
)


pipeline = regression_training_with_pos_pipeline(
    dataset_config, smiles, regression_targets, regression_masks
) 

#pipeline = regression_training_with_transition_probs_pipeline(
#    dataset_config, smiles, regression_targets, regression_masks
#) 

dataset = pipeline.build()


store_data_to_disk(dataset, f"{dataset_directory}_full")
#
from threedscriptors.data_handling.data_utils import get_atom_species_in_smiles
##
atom_type_set = get_atom_species_in_smiles(ListSmilesIterator(dataset.smiles_list))

dataset_split = [DatasetSplit.TRAIN, DatasetSplit.VALIDATION]
splitting_ratio = [0.8,0.2]
#
split_datasets = dataset.split_dataset(splitting_ratio, dataset_split)
##

train_dataset = split_datasets[0]
store_data_to_disk(train_dataset, f"{dataset_directory}_train")

validation_dataset = split_datasets[1]
store_data_to_disk(validation_dataset, f"{dataset_directory}_valid")


DatasetPostLoadAnalysis(dataset, f"{dataset_directory}_full").run()
DatasetPostLoadAnalysis(train_dataset, f"{dataset_directory}_train").run()
DatasetPostLoadAnalysis(validation_dataset,f"{dataset_directory}_valid").run()

print(dataset.embeddings.element_size() * dataset.embeddings.nelement())