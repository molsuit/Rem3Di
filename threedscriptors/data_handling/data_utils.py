import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import rdkit.Chem as Chem
import torch
from ase import Atoms
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Chem import rdmolops
from rdkit2ase import rdkit2ase


if TYPE_CHECKING:
    from threedscriptors.configuration.data_config import DatasetConfig, TaskConfig
from threedscriptors.data_handling.smiles_iterator import SmilesIterator
from threedscriptors.model.transformer_components import TransformerEncoder


def get_mirrored_molecules(molecules: list[Atoms]):
    mirrored_molecules = []

    for mol in molecules:
        mirrored_mol = mol.copy()
        mirrored_mol.set_positions(-mol.get_positions())
        mirrored_molecules.append(mirrored_mol)

    return mirrored_molecules


def relax_atoms(atoms: Atoms, calculator: MACECalculator, BFGS_tol=0.05, max_steps=100):
    atoms.calc = calculator
    dyn = LBFGS(atoms, logfile=None)
    converged = dyn.run(fmax=BFGS_tol, steps=max_steps)
    if not converged:
        raise ValueError("LBFGS did not converge")


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

    atoms = rdkit2ase(mol)
    return atoms


def get_ase_atoms_with_conformers(smiles, N_conformers: int, load_adjacency_matrix) -> list[Atoms]:
    # print(smiles)
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError

    mol = Chem.AddHs(mol)
    charge = Chem.GetFormalCharge(mol)
    if charge != 0:
        raise ValueError("Charged molecule")

    EmbedMultipleConfs(
        mol, numConfs=N_conformers, numThreads=N_conformers, maxAttempts=5000
    )

    if load_adjacency_matrix:
        A = rdmolops.GetAdjacencyMatrix(mol)


        ase_confs = [
            Atoms(
                positions=conf.GetPositions(),
                numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],info = {"adjacency_matrix" : A}
            )
            for conf in mol.GetConformers()
        ]

    else:
        ase_confs = [
            Atoms(
                positions=conf.GetPositions(),
                numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()]
            )
            for conf in mol.GetConformers()
        ]

    

    return ase_confs, mol


def get_relaxed_conformers(
    smiles,
    mace_calculator: MACECalculator,
    dataset_config: "DatasetConfig",
    N_conformers: int,
):
    if N_conformers == 1:
        confs = [get_ase_atoms(smiles)]
    else:
        confs = get_ase_atoms_with_conformers(smiles, N_conformers)

    molecules = []

    for atoms in confs:
        try:
            relax_atoms(
                atoms,
                mace_calculator,
                dataset_config.BFGS_tol,
                dataset_config.BFGS_max_steps,
            )
        except ValueError:
            print("Molecule did not relax.")
            continue
        else:
            molecules.append(atoms)
    return molecules


def get_max_molecule_size_from_smiles(
    smiles_iterator: SmilesIterator, max_num_molecules=math.inf
) -> int:
    max_atoms = 0
    for i, smiles in enumerate(smiles_iterator):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        if num_atoms > max_atoms:
            max_atoms = num_atoms
        if i >= max_num_molecules:
            break
    return max_atoms


def get_max_molecule_size_from_atoms(atoms: list[Atoms]):
    max_atoms = 0
    for mol in atoms:
        number_of_atoms = len(mol)
        max_atoms = max(max_atoms, number_of_atoms)
    return max_atoms

def get_max_num_of_heavy_atom_from_atoms(atoms : list[Atoms]):

    heavy_counts = ((mol.get_atomic_numbers() != 1).sum() for mol in atoms)
    return max(heavy_counts, default=0)



def get_atom_species_in_smiles(smiles_iterator: SmilesIterator):
    atom_species_set = set("H")
    for smiles in smiles_iterator:
        mol = Chem.MolFromSmiles(smiles)
        atom_species_set.update([atom.GetSymbol() for atom in mol.GetAtoms()])
    return atom_species_set


def get_global_descriptor(
    smiles: str, encoder: TransformerEncoder, calculator: MACECalculator
):
    # TODO: Maybe check SMILES validity?
    atoms = get_ase_atoms(smiles)
    mace_des = calculator.get_descriptors(atoms, invariants_only=True)
    mace_des = torch.tensor(mace_des).unsqueeze(0).float()
    encoder.eval()
    with torch.no_grad():
        global_descriptor = encoder(mace_des)

    return global_descriptor


def has_task_with_auxillary_data(tasks: Sequence["TaskConfig"]) -> bool:
    for task in tasks:
        if task.has_auxillary_data:
            return True

    return False


def get_unique_smiles_id_from_smiles_list(smiles_list: list[str]):
    string_to_id = {}
    result_ids = []
    current_id = 0

    # Process each string in the list
    for s in smiles_list:
        if s not in string_to_id:
            # Assign a new integer if the string has not been seen before
            string_to_id[s] = current_id
            current_id += 1
        # Append the mapped integer
        result_ids.append(string_to_id[s])

    return result_ids


def get_functional_group_label(smiles: list[str]):
    # This function is specific to the test functional group dataset, and is not meaningful in any other context.

    functional_group_indices = {"OH": [], "NH2": [], "SH": []}
    # Conformers???
    for smiles_index, smiles_string in enumerate(smiles):
        match smiles_string[0]:
            case "O":
                functional_group_indices["OH"].append(smiles_index)
            case "S":
                functional_group_indices["SH"].append(smiles_index)
            case "N":
                functional_group_indices["NH2"].append(smiles_index)
            case _:
                raise ValueError("Non matching smiles in functional group dataset")

    return functional_group_indices


def validate_ratios(ratios: Sequence[float]) -> None:
    """Ensure the ratios add up to 1 (within 1 e-6) and are all positive."""
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"`ratios` must sum to 1 (got {ratios!r})")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be strictly positive")




def compute_splits(size: int, ratios: Sequence[float], split_interval) -> list[slice]:
    """Return slice objects for each split boundary."""
    raw_counts = (np.asarray(ratios) * size).astype(int)

    base = (raw_counts // split_interval).astype(int) * split_interval

    leftover = (size - base.sum()) // split_interval

    base[0] += leftover* split_interval

    # Fix any rounding drift so the slices cover the full length

    offsets = np.cumsum(np.insert(base, 0, 0))

    return [slice(offsets[i], offsets[i + 1]) for i in range(len(ratios))]


def rmsd(A, B):
    """
    Compute RMSD between two point sets A and B (shape: Nx3).
    Centers each, finds optimal rotation, then returns RMSD.
    """
    # center
    A_cent = A - A.mean(axis=0)
    B_cent = B - B.mean(axis=0)

    # covariance
    C = A_cent.T @ B_cent

    # SVD
    V, S, Wt = np.linalg.svd(C)

    # ensure right‐handed coordinate system
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1,1,d]) @ Wt

    # rotated A and RMSD
    A_rot = A_cent @ U
    return np.sqrt(((A_rot - B_cent)**2).sum() / A.shape[0])