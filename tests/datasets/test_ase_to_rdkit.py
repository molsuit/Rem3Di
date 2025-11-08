import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem.rdchem import Conformer
from rdkit.Geometry import Point3D

from threedscriptors.data_handling.data_utils import get_rdkit_mol_from_ase  # replace with actual import



def get_ase_from_rdkit(mol: Chem.Mol) -> Atoms:
    """Helper: convert RDKit Mol with conformer back to ASE Atoms."""
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return Atoms(numbers=numbers, positions=coords)

def test_roundtrip_rdkit_ase():
    # Original ASE object
    original_atoms = Atoms(
        numbers=[6, 6, 8, 1, 1],
        positions=[
            (0.000,  0.000,  0.000),
            (1.400,  0.000,  0.000),
            (2.100,  1.100,  0.000),
            (-0.550,  0.950,  0.000),
            (-0.550, -0.950,  0.000),
        ]
    )

    # ASE → RDKit
    mol = get_rdkit_mol_from_ase(original_atoms)

    # RDKit → ASE
    roundtrip_atoms = get_ase_from_rdkit(mol)

    # Compare atomic numbers
    assert np.array_equal(original_atoms.get_atomic_numbers(),
                          roundtrip_atoms.get_atomic_numbers())

    # Compare positions (allowing tiny floating-point error)
    assert np.allclose(original_atoms.get_positions(),
                       roundtrip_atoms.get_positions(), atol=1e-6)
