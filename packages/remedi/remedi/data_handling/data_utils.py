from collections.abc import Sequence

import numpy as np
import rdkit.Chem as Chem
from ase import Atoms
from rdkit.Chem import AllChem


def get_ase_atoms(smiles) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    returncode = AllChem.EmbedMolecule(
        mol, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, randomSeed=-1
    )

    if returncode == -1:
        AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            randomSeed=-1,
        )

    conf = mol.GetConformer()
    atoms = Atoms(
        positions=conf.GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        info={"smiles": smiles},
    )

    return atoms


def get_functional_group_label(smiles: list[str]):
    # This function is specific to the test functional group dataset, and is not meaningful in any other context.
    print(smiles)
    functional_group_indices = {"OH": [], "NH2": [], "SH": []}
    # Conformers???
    for smiles_index, smiles_string in enumerate(smiles):
        match smiles_string[-1]:
            case "O":
                functional_group_indices["OH"].append(smiles_index)
            case "S":
                functional_group_indices["SH"].append(smiles_index)
            case "N":
                functional_group_indices["NH2"].append(smiles_index)
            case _:
                print(f"Smi {smiles_string}")
                raise ValueError("Non matching smiles in functional group dataset")

    return functional_group_indices


def compute_splits(size: int, ratios: Sequence[float]) -> list[slice]:
    """Return slice objects for each split boundary."""
    raw_counts = (np.asarray(ratios) * size).astype(int)

    leftover = size - raw_counts.sum()

    raw_counts[0] += leftover

    # Fix any rounding drift so the slices cover the full length

    offsets = np.cumsum(np.insert(raw_counts, 0, 0))

    return [slice(offsets[i], offsets[i + 1]) for i in range(len(ratios))]
