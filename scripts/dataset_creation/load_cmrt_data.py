from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
from threedscriptors.data_handling.dataset import RegressionWithAuxDataset, RegressionWithAuxAndPositionsDataset
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import chiral_regression_training_pipeline
from threedscriptors.data_handling.source_preprocessing.cmrt_preprocessing import (
    load_cmrt_data,
)

dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/cmrt"
)

smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data(
    f"{dataset_directory}/raw_data.csv", single_column_type=True
)

print(aux_data)



MACE_PATH = (
    "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
)
print(len(smiles))
embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda", default_dtype="float64"),
    model_name="mace_off_24_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)
dataset_config = DatasetConfig(
    N_molecules=20000,
    dataset_type=RegressionWithAuxAndPositionsDataset,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=2,
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


print(dataset.regression_masks)
print(dataset.regression_targets)



store_data_to_disk(dataset, f"{dataset_directory}_full")

from threedscriptors.data_handling.data_utils import get_atom_species_in_smiles
atom_type_set = get_atom_species_in_smiles(ListSmilesIterator(dataset.smiles_list))
print(atom_type_set)


splitting_ratio = [0.8,0.2]

split_datasets = dataset.split_dataset(splitting_ratio)

store_data_to_disk(split_datasets[0], f"{dataset_directory}_train")
store_data_to_disk(split_datasets[1], f"{dataset_directory}_valid")

