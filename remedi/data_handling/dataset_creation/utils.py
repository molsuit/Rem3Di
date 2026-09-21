from time import perf_counter

import numpy as np
import torch
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from torch import Tensor

from remedi.data_handling.dataset_creation.conformer_timing import (
    ConformerTimingRecord,
    EmbedResult,
)


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
    mmff_non_bonded_thresh: float = 100.0,
) -> EmbedResult:
    """Embed + MMFF-relax one molecule, returning a fully populated EmbedResult.

    **Never raises.** Parse / embed / MMFF failures are encoded in the embedded
    ``ConformerTimingRecord.status`` so the caller can record the partial-phase
    timing even when the result is unusable. That includes the exceptions RDKit
    throws from inside its own C++ (``RuntimeError: Invariant Violation``
    out of the BFGS line search is the one seen in production): this function
    runs one molecule of a panel of thousands, and a raise here propagates out
    of the worker pool and loses the whole dataset's prepare task.
    """
    n_atoms = -1
    t_embed_s = 0.0
    t_mmff_s = 0.0

    def _failure(
        status,
        error_msg: str,
        nonisomeric: str | None = None,
        n_emitted: int = 0,
    ) -> EmbedResult:
        return EmbedResult(
            nonisomeric_smiles=nonisomeric,
            positions=None,
            atomic_numbers=None,
            timing=ConformerTimingRecord(
                isomeric_smiles=isomeric_smiles,
                n_atoms=n_atoms,
                n_confs_requested=int(n_confs),
                n_confs_emitted=n_emitted,
                t_embed_s=t_embed_s,
                t_mmff_s=t_mmff_s,
                status=status,
                error_msg=error_msg,
            ),
        )

    try:
        parsed = Chem.MolFromSmiles(isomeric_smiles)
        if parsed is None:
            return _failure("value_error", f"Bad SMILES: {isomeric_smiles}")
        nonisomeric = Chem.MolToSmiles(parsed, isomericSmiles=False)
        mol = Chem.AddHs(parsed)
    except (RuntimeError, ValueError) as error:
        return _failure(
            "value_error", f"RDKit raised parsing {isomeric_smiles}: {error}"
        )
    n_atoms = mol.GetNumAtoms()

    params = AllChem.ETKDGv3()
    params.numThreads = 1
    params.maxIterations = int(max_embed_attempts)
    params.useRandomCoords = False
    params.useSmallRingTorsions = True
    params.useExpTorsionAnglePrefs = True
    params.enforceChirality = True

    t0 = perf_counter()
    used_random_coordinates = False
    try:
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params)
        if not ids:
            # Bridged ring systems, quaternary centres and macrocycles routinely
            # defeat the distance-geometry start; random initial coordinates are
            # RDKit's documented fallback and recover most of them.
            params.useRandomCoords = True
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params)
            used_random_coordinates = True
    except (RuntimeError, ValueError) as error:
        # RDKit signals an internal invariant violation as a bare RuntimeError.
        # One pathological molecule must not sink a whole dataset's task.
        t_embed_s = perf_counter() - t0
        return _failure(
            "embed_failed",
            f"RDKit raised during embedding of {isomeric_smiles}: {error}",
            nonisomeric=nonisomeric,
        )
    t_embed_s = perf_counter() - t0

    if not ids:
        return _failure(
            "embed_failed",
            f"No conformers embedded for {isomeric_smiles}",
            nonisomeric=nonisomeric,
        )

    t0 = perf_counter()
    try:
        AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            maxIters=int(max_opt_iters),
            nonBondedThresh=float(mmff_non_bonded_thresh),
            numThreads=1,
        )
    except (RuntimeError, ValueError) as error:
        # Observed in production as
        # ``RuntimeError: Invariant Violation / bad direction in linearSearch
        # ... BFGSOpt.h``; it cost three datasets their whole generate_conformers
        # task before this guard.
        t_mmff_s = perf_counter() - t0
        return _failure(
            "mmff_failed",
            f"RDKit raised during MMFF relaxation of {isomeric_smiles}: {error}",
            nonisomeric=nonisomeric,
        )
    t_mmff_s = perf_counter() - t0

    atomic_numbers = np.array(
        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64
    )
    confs = mol.GetConformers()
    if len(confs) == 0:
        return _failure(
            "mmff_failed",
            f"No conformers after MMFF for {isomeric_smiles}",
            nonisomeric=nonisomeric,
        )

    positions = np.stack(
        [conf.GetPositions().astype(np.float64, copy=False) for conf in confs], axis=0
    )
    n_emitted = positions.shape[0]

    return EmbedResult(
        nonisomeric_smiles=nonisomeric,
        positions=positions,
        atomic_numbers=atomic_numbers,
        timing=ConformerTimingRecord(
            isomeric_smiles=isomeric_smiles,
            n_atoms=n_atoms,
            n_confs_requested=int(n_confs),
            n_confs_emitted=n_emitted,
            t_embed_s=t_embed_s,
            t_mmff_s=t_mmff_s,
            status="ok",
            used_random_coordinates=used_random_coordinates,
        ),
    )


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
