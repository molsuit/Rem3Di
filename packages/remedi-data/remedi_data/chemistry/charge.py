"""Total charge and spin multiplicity of a molecule, as a dataset table stores them."""

from __future__ import annotations

from rdkit import Chem


def charge_and_multiplicity(molecule: Chem.Mol | None) -> tuple[float, float]:
    """``(total formal charge, spin multiplicity)`` of an RDKit molecule.

    Multiplicity is ``2S + 1`` with ``2S`` the number of unpaired (radical)
    electrons, so a closed-shell molecule is 1.0. Explicit hydrogens do not
    change either value.

    Raises:
        ValueError: if ``molecule`` is ``None`` (RDKit could not parse it).
    """
    if molecule is None:
        raise ValueError("no molecule: RDKit could not parse it")
    unpaired = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    return float(Chem.GetFormalCharge(molecule)), float(unpaired + 1)
