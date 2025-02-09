import matplotlib.pyplot as plt
import numpy as np
from ase.optimize import LBFGS
from ase.visualize.plot import plot_atoms
from e3nn import o3
from mace.calculators import mace_mp
from mace.modules.blocks import tp_out_irreps_with_instructions

from threedscriptors.data_handling.preprocessing import get_ase_atoms

smiles = "CC(N)O"
atoms = get_ase_atoms(smiles)
fig = plt.figure(1)
ax = fig.gca()
plot_atoms(atoms, ax=ax)
fig.savefig("test.png")

calc = mace_mp("medium", "cuda", enable_cueq=True)
atoms.calc = calc
opt = LBFGS(atoms)
opt.run(fmax=0.1, steps=10)

pos = atoms.get_positions()
atoms2 = atoms.copy()
pos2 = pos * -1.0
atoms2.set_positions(pos2)
des = calc.get_descriptors(atoms, invariants_only=False, num_layers=1)
des2 = calc.get_descriptors(atoms2, invariants_only=False, num_layers=1)

print(des)
print(np.sum(des - des2))
print(des.shape)


print(des[1, :].shape)

# These are the node features that are present in the macemp0 medium model
i_in1 = o3.Irreps("128x0e + 128x1o")

# not full tensor product
tp_1 = o3.FullTensorProduct(i_in1, i_in1)


fig = plt.figure()


# They create intermediate even vectors, that are combined with the odd vectors to produce
lin = o3.Linear(tp_1.irreps_out, "10x1e")
print(lin)

irreps_mid, instructions = tp_out_irreps_with_instructions(
    i_in1,
    lin.irreps_out,
    o3.Irreps("128x0o"),
)

tp_2 = o3.TensorProduct(
    i_in1,
    lin.irreps_out,
    irreps_mid,
    instructions=instructions,
    shared_weights=True,
    internal_weights=True,
)

print(tp_2)
