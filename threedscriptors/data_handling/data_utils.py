import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import rdkit.Chem as Chem
from ase import Atoms
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
from rdkit.Chem import AllChem, rdDetermineBonds, rdmolops
from rdkit.Chem.rdchem import Conformer
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from rdkit.Geometry import Point3D

if TYPE_CHECKING:
    from threedscriptors.configuration.data_config import DatasetConfig, TaskConfig


def get_molecular_weight(molecules: list[Atoms]):
    return [sum(m.get_masses()) for m in molecules]


def get_mirrored_molecules(molecules: list[Atoms], new_smiles: str):
    mirrored_molecules = []

    for mol in molecules:
        mirrored_mol = mol.copy()
        mirrored_mol.set_positions(-mol.get_positions())
        mirrored_mol.info["smiles"] = new_smiles
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

    conf = mol.GetConformer()
    atoms = Atoms(
        positions=conf.GetPositions(),
        numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        info={"smiles": smiles},
    )

    return atoms


def contains_ionic_atom(mol: Chem.Mol) -> bool:
    """Return True if *any* atom in `mol` has non-zero formal charge."""
    return any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def get_ase_atoms_with_conformers(smiles, N_conformers: int) -> list[Atoms]:
    # print(smiles)
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError

    mol = Chem.AddHs(mol)
    charged = contains_ionic_atom(mol)

    if charged:
        raise ValueError(f"Smiles {smiles} is a charged molecule")

    EmbedMultipleConfs(
        mol, numConfs=N_conformers, numThreads=N_conformers, maxAttempts=500
    )

    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, nonBondedThresh=500.0)

    ase_confs = [
        Atoms(
            positions=conf.GetPositions(),
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
            info={"smiles": smiles},
        )
        for conf in mol.GetConformers()
    ]

    if len(ase_confs) == 0:
        raise ValueError

    return ase_confs


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


def count_atoms_from_smiles(
    smiles_iterator, heavy_atoms_only=False, max_num_molecules=np.inf
) -> int:
    # Returns the max and sum of the atoms from smiles

    atom_count = []

    for i, smiles in enumerate(smiles_iterator):
        mol = Chem.MolFromSmiles(smiles)

        if not heavy_atoms_only:
            mol = Chem.AddHs(mol)

        atom_count.append(mol.GetNumAtoms())

        if i >= max_num_molecules:
            break

    return max(atom_count, default=0), sum(atom_count)


def get_all_atom_counts(atoms: list[Atoms], heavy_atoms_only=False):
    if heavy_atoms_only:
        counts = ((mol.get_atomic_numbers() != 1).sum() for mol in atoms)
    else:
        counts = (len(mol) for mol in atoms)

    return counts


def count_atoms_from_ase(atoms: list[Atoms], heavy_atoms_only=False):
    # Returns the max and the sum of the numbers of atoms inside a list of ase atoms
    counts = get_all_atom_counts(atoms, heavy_atoms_only)

    return max(counts, default=0), sum(counts)


def get_atom_species_in_smiles(smiles_iterator):
    atom_species_set = set("H")
    for smiles in smiles_iterator:
        mol = Chem.MolFromSmiles(smiles)
        atom_species_set.update([atom.GetSymbol() for atom in mol.GetAtoms()])
    return atom_species_set


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


def validate_ratios(ratios: Sequence[float]) -> None:
    """Ensure the ratios add up to 1 (within 1 e-6) and are all positive."""
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"`ratios` must sum to 1 (got {ratios!r})")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be strictly positive")


def compute_splits(size: int, ratios: Sequence[float]) -> list[slice]:
    """Return slice objects for each split boundary."""
    raw_counts = (np.asarray(ratios) * size).astype(int)

    leftover = size - raw_counts.sum()

    raw_counts[0] += leftover

    # Fix any rounding drift so the slices cover the full length

    offsets = np.cumsum(np.insert(raw_counts, 0, 0))

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

    # ensure right-handed coordinate system
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1, 1, d]) @ Wt

    # rotated A and RMSD
    A_rot = A_cent @ U
    return np.sqrt(((A_rot - B_cent) ** 2).sum() / A.shape[0])


def get_rdkit_mol_from_ase(
    atoms: Atoms,
    *,
    assign_bonds: bool = False,
    charge: int = 0,
    cov_factor: float = 1.3,
    use_hueckel: bool = True,
) -> Chem.Mol:
    """
    Build an RDKit Mol from ASE Atoms using only atomic numbers + 3D coordinates.

    Parameters
    ----------
    atoms : ASE Atoms
        Source object. Positions must be (N, 3).
    assign_bonds : bool
        If True, use RDKit's distance/valence heuristic to infer connectivity.
    charge : int
        Total molecular charge to use when inferring bonds.
    cov_factor : float
        Distance cutoff scaling for bond inference (default 1.3).
    use_hueckel : bool
        Use Hückel test during bond inference (helps aromatics).

    Returns
    -------
    rdkit.Chem.Mol
        Molecule with one 3D conformer attached. Not sanitized.
    """
    n = len(atoms)
    if n == 0:
        return Chem.Mol()

    coords = np.asarray(atoms.get_positions(), dtype=float)
    zs: Sequence[int] = np.asarray(atoms.get_atomic_numbers(), dtype=int).tolist()

    if coords.shape != (n, 3):
        raise ValueError(f"positions must be shape ({n}, 3), got {coords.shape}")

    rw = Chem.RWMol()
    for Z in zs:
        rw.AddAtom(Chem.Atom(int(Z)))

    conf = Conformer(n)
    conf.Set3D(True)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    rw.AddConformer(conf, assignId=True)

    mol = rw.GetMol()

    if assign_bonds:
        rdDetermineBonds.DetermineBonds(
            mol,
            charge=charge,
            covFactor=cov_factor,
            useHueckel=use_hueckel,
        )
        # Optional: sanitize if you want strict valence checks
        # Chem.SanitizeMol(mol)

    return mol


def mol_is_fragmented(mol, *, ignore_hs=True) -> bool:
    m = Chem.RemoveHs(mol) if ignore_hs else mol
    # tuple of tuples of atom indices per fragment
    frags = rdmolops.GetMolFrags(m, asMols=False)
    return len(frags) > 1
