from dataset import AtomEmbeddingDataset
from mace.calculators import MACECalculator
from SmilesIterator import SmilesIterator

MACE_PATH = "/home/steffen/projects/mol_descriptors/mace_model/2023-12-10-mace-128-L0_energy_epoch-249.model"

mace_calculator = MACECalculator(model_path=MACE_PATH, device='cuda',enable_cueq = True)


smiles_iterator = SmilesIterator(source="/home/steffen/projects/mol_descriptors/data/GDB-13s_shuffled.smi")
AtomEmbeddingDataset.construct_from_smiles(iterator = smiles_iterator,mace_caluclator=mace_calculator, N_molecules=2048, BFGS_tol=0.1, max_atoms=3*13 + 2, embedding_size=256)