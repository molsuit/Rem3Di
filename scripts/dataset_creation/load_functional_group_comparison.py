from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import pretraining_pipeline
from threedscriptors.data_handling.smiles_iterator import FileSmilesIterator

directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/functional_group_dataset"
smiles_file = f"{directory}/smiles_list"

smiles_list = list(FileSmilesIterator(smiles_file))



MACE_PATH = (
    "/share/snw30/projects/mace_model/mace_agnesi_medium.model"
)

embedding_model_config = MaceCalculatorConfig(
    mace_calc=None,
    model_name="medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)


dataset_config = DatasetConfig(
    N_molecules=None,
    dataset_type=DatasetTypes.PRETRAINING,
    BFGS_tol=0.001,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
)

dataset = pretraining_pipeline(dataset_config, smiles_list).build()
store_data_to_disk(dataset, directory)
