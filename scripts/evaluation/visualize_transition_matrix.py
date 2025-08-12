import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

import matplotlib.pyplot as plt


np.set_printoptions(linewidth=np.inf)

def transition_matrix_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    A = rdmolops.GetAdjacencyMatrix(mol)
    A_self = A + np.eye(A.shape[0])
    D_inv = np.diag(1.0 / A_self.sum(1))

    return torch.tensor(D_inv @ A_self, dtype=torch.float32), mol

T, mol = transition_matrix_from_smiles("CC(C(=O)O)N")   # your example

T = 0.5*(T+T.T)


print(T.numpy())
plt.figure(figsize=(2,2))
plt.imshow(T.numpy())
plt.axis("off")
plt.show()
plt.savefig("transition_matrix.svg")
