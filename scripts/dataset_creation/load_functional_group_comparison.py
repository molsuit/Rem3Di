from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset import AtomicEmbeddingWithPositionsDataset
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import pretraining_pipeline_with_positions
from threedscriptors.data_handling.smiles_iterator import FileSmilesIterator

directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/functional_group_dataset"
smiles_file = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/functional_group_smiles_list"

smiles_list = list(FileSmilesIterator(smiles_file))



MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"



embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_off_24_medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)

dataset_config = DatasetConfig(
    N_molecules=len(smiles_list),
    dataset_type=AtomicEmbeddingWithPositionsDataset,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=None,
    only_heavy_atoms=False,
    dataset_name="functional_groups",
)

dataset = pretraining_pipeline_with_positions(dataset_config, smiles_list).build()
store_data_to_disk(dataset, directory)
