import numpy.testing as npt
import torch
from mace.calculators import mace_mp

from threedscriptors.data_handling.data_utils import get_ase_atoms
from threedscriptors.model.atomic_descriptor_preprocess import PseudoscalarGenerator
from threedscriptors.utils.model_utils import get_mace_calculator_irrep_signature

smiles = "CC(N)O"
atoms = get_ase_atoms(smiles)

calc = mace_mp("medium", "cuda", enable_cueq=True)

pos = atoms.get_positions()
atoms2 = atoms.copy()
pos2 = pos * -1.0
atoms2.set_positions(pos2)

des1 = calc.get_descriptors(atoms, invariants_only=False)

des2 = calc.get_descriptors(atoms2, invariants_only=False)

sig = get_mace_calculator_irrep_signature(calc)
print(sig)


ps_generator = PseudoscalarGenerator()

des_ps = ps_generator.forward(torch.Tensor(des1) * 10)
des_ps2 = ps_generator.forward(torch.Tensor(des2) * 10)

print(des_ps.shape)


npt.assert_array_almost_equal(des_ps.detach().numpy(), -des_ps2.detach().numpy())
