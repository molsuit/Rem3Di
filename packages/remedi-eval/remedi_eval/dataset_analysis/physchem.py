"""The cheap 2D RDKit descriptors the dataset analysis reports per molecule.

``DESCRIPTORS_2D`` maps a name to a function of a SMILES-derived ``Chem.Mol``
(no geometry needed).
"""

from __future__ import annotations

from collections.abc import Callable

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors
from rdkit.Chem import rdMolDescriptors as rdMD

DESCRIPTORS_2D: dict[str, Callable[[Chem.Mol], float]] = {
    "mw": lambda m: float(Descriptors.MolWt(m)),
    "n_heavy_atoms": lambda m: float(m.GetNumHeavyAtoms()),
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "tpsa": lambda m: float(rdMD.CalcTPSA(m)),
    "hbd": lambda m: float(rdMD.CalcNumHBD(m)),
    "hba": lambda m: float(rdMD.CalcNumHBA(m)),
    "rot_bonds": lambda m: float(rdMD.CalcNumRotatableBonds(m)),
    "fraction_csp3": lambda m: float(rdMD.CalcFractionCSP3(m)),
    "n_rings": lambda m: float(rdMD.CalcNumRings(m)),
    "n_aromatic_rings": lambda m: float(rdMD.CalcNumAromaticRings(m)),
    "qed": lambda m: float(QED.qed(m)),
    "mol_mr": lambda m: float(Crippen.MolMR(m)),
}
