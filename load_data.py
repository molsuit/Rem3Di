from dataset import AtomEmbeddingDataset
from mace.calculators import MACECalculator


MACE_PATH = "/home/steffen/projects/mol_descriptors/mace_model/2023-12-10-mace-128-L0_energy_epoch-249.model"

mace_calculator = MACECalculator(model_path=MACE_PATH, device='cuda')

AtomEmbeddingDataset.construct_from_smiles(smiles_path="/home/steffen/projects/mol_descriptors/data/GDB-13s.smi",mace_caluclator=mace_calculator, N_molecules=250, BFGS_tol=0.5, max_atoms=3*13 + 2, embedding_size=256)