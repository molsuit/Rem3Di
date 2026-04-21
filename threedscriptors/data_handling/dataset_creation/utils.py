import numpy as np
import torch
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from torch import Tensor


def system_idx_to_ragged_ptr(system_idx: Tensor) -> Tensor:
    """
    Convert a per-atom `system_idx` vector into a ragged pointer (cumulative end
    indices), optionally forcing the pointer length to be `n_mols`.

    Using `minlength=n_mols` ensures we include trailing zero-count systems when
    the last system(s) contributed no atoms, keeping lengths consistent with
    metadata like `structure_ids`.
    """

    counts = torch.bincount(system_idx)
    ptr = torch.cumsum(counts, dim=0)
    return ptr


def ensure_numpy_array(array: Tensor | np.ndarray | None):
    if array is None:
        return None

    array = (
        array.detach().cpu().numpy()
        if isinstance(array, torch.Tensor)
        else np.asarray(array)
    )

    return array


def embed_one_smiles(
    isomeric_smiles: str,
    n_confs: int,
    max_embed_attempts: int,
    max_opt_iters: int,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """
    Returns:
        isomeric_smiles,
        nonisomeric_smiles,
        positions: (K, A, 3) float64
        atomic_numbers: (A,) int64
    Raises:
        ValueError if embedding fails or zero conformers produced.
    """
    # Build non-isomeric SMILES for convenience (fast to recompute)
    nonisomeric = Chem.MolToSmiles(
        Chem.MolFromSmiles(isomeric_smiles), isomericSmiles=False
    )

    mol = Chem.MolFromSmiles(isomeric_smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {isomeric_smiles}")

    mol = Chem.AddHs(mol)

    # ETKDGv3 with deterministic seeding if provided
    params = AllChem.ETKDGv3()
    # print(params.keys)
    params.numThreads = 1
    params.maxIterations = int(max_embed_attempts)
    # Some quality-of-life flags that help with odd chemistries
    params.useRandomCoords = False
    params.useSmallRingTorsions = True
    params.useExpTorsionAnglePrefs = True
    params.enforceChirality = True

    ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params)
    if not ids:
        raise ValueError(f"No conformers embedded for {isomeric_smiles}")

    # MMFF optimize (UFF as fallback could be added)
    AllChem.MMFFOptimizeMoleculeConfs(
        mol,
        maxIters=int(max_opt_iters),
        nonBondedThresh=500.0,  # generous so we don't drop too many
        numThreads=1,
    )
    # We don't filter on convergence by default; you can filter here if desired

    atomic_numbers = np.array(
        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64
    )
    confs = mol.GetConformers()
    positions = np.stack(
        [conf.GetPositions().astype(np.float64, copy=False) for conf in confs], axis=0
    )  # (K, A, 3)

    if positions.shape[0] == 0:
        raise ValueError(f"No ASE conformers for {isomeric_smiles}")

    return isomeric_smiles, nonisomeric, positions, atomic_numbers


def get_ase_atoms_with_conformers(smiles, N_conformers: int) -> list[Atoms]:
    # print(smiles)
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError

    mol = Chem.AddHs(mol)

    EmbedMultipleConfs(
        mol, numConfs=N_conformers, numThreads=N_conformers, maxAttempts=500
    )

    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, nonBondedThresh=500.0)

    ase_confs = [
        Atoms(
            positions=conf.GetPositions(),
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
        )
        for conf in mol.GetConformers()
    ]

    if len(ase_confs) == 0:
        raise ValueError

    return ase_confs
