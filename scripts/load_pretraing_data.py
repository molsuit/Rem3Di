from mace.calculators import MACECalculator

from threedscriptors.data_handling.dataset import DatasetFactory
from threedscriptors.data_handling.smiles_iterator import FileSmilesIterator

MACE_PATH = "/home/steffen/projects/mol_descriptors/mace_model/2023-12-10-mace-128-L0_energy_epoch-249.model"

mace_calculator = MACECalculator(model_path=MACE_PATH, device="cuda", enable_cueq=True)

metadata = {
    "N_molecules": 2048,
    "BFGS_tol": 0.1,
    "max_atoms": 3 * 13 + 2,
    "embedding_size": 256,
    "BFGS_max_steps": 200,
    "dataset_type": "Pretraining",
}

smiles_iterator = FileSmilesIterator(
    "/home/steffen/projects/mol_descriptors/data/gdb13/GDB-13s_shuffled.smi"
)

dataset = DatasetFactory.from_smiles(
    iterator=smiles_iterator, mace_caluclator=mace_calculator, metadata=metadata
)
dataset.store_data_to_disk("/home/steffen/projects/mol_descriptors/data/gdb13/")
