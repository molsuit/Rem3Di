from rdkit import Chem

MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def filter_mol(mol: Chem.Mol, require_3D=False, max_atoms: int | None = None) -> bool:
    """Return True if mol passes all filters, otherwise False."""
    try:
        if mol is None:
            return False
        # must have a 3D conformer
        if require_3D and (mol.GetNumConformers() == 0):
            return False
        # single fragment only
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 3:
            return False
        if max_atoms is not None and num_atoms > max_atoms:
            return False
        if (
            Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
            and len(Chem.GetMolFrags(mol, asMols=True)) > 1
        ):
            return False

        for a in mol.GetAtoms():
            if a.GetSymbol() not in MACE_OFF_ELEMENTS:
                return False
            if a.GetIsotope() != 0:
                return False
        return True
    except Exception:
        return False
