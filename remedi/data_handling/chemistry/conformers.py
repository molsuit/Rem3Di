"""Conformer embedding: ETKDGv3 + MMFF94, one molecule at a time or a pool of them.

:func:`embed_one_smiles` never raises: every parse / embed / relaxation failure
becomes a :class:`ConformerTimingRecord` with a failure status, so one
pathological molecule cannot sink a dataset of thousands. :func:`embed_many`
runs it over a process pool and also survives a worker process that dies (a
segfault inside RDKit, the OOM killer): the molecules that were in flight are
retried, and a molecule that keeps killing its worker is isolated and recorded
as ``worker_crashed``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Hashable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class ConformerEmbeddingConfig(BaseModel):
    """The ETKDG + MMFF settings.

    The defaults are the validated production values: ``max_embed_attempts``
    is RDKit's ``params.maxIterations``, and 200 covers essentially anything
    ETKDG can embed while larger values only inflate the wall-time tail on
    pathological molecules (CYP timing experiment, slurm-4686108: 5.2 h to
    10 min at identical ~99 % yield). ``mmff_non_bonded_threshold`` is in
    Angstrom; RDKit's default 100.0 already includes every atom pair of a
    drug-sized molecule.
    """

    model_config = ConfigDict(extra="forbid")

    n_conformers: int = Field(default=1, ge=1)
    max_embed_attempts: int = Field(default=200, ge=1)
    max_mmff_steps: int = Field(default=100, ge=0)
    mmff_non_bonded_threshold: float = Field(default=100.0, gt=0.0)


ConformerTimingStatus = Literal[
    "ok",
    "embed_failed",
    "mmff_failed",
    "value_error",
    "other_error",
    "worker_crashed",
]


class ConformerTimingRecord(BaseModel):
    """Wall time and outcome of embedding one molecule."""

    model_config = ConfigDict(extra="forbid")

    isomeric_smiles: str
    n_atoms: int
    n_confs_requested: int
    n_confs_emitted: int
    t_embed_s: float
    t_mmff_s: float
    status: ConformerTimingStatus
    error_msg: str | None = None
    # True when the plain ETKDG start produced nothing and the conformer came
    # from the random-initial-coordinates retry.
    used_random_coordinates: bool = False


@dataclass
class EmbedResult:
    """What :func:`embed_one_smiles` returns: always populated, never raised."""

    nonisomeric_smiles: str | None
    #: ``(n_conformers, n_atoms, 3)``, or ``None`` on failure.
    positions: np.ndarray | None
    atomic_numbers: np.ndarray | None
    timing: ConformerTimingRecord

    @property
    def succeeded(self) -> bool:
        return (
            self.timing.status == "ok"
            and self.positions is not None
            and self.atomic_numbers is not None
        )


def write_timings_jsonl(records: list[ConformerTimingRecord], path: Path) -> None:
    """Truncate-write the timing records as JSONL (one record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for record in records:
            file.write(record.model_dump_json() + "\n")


def _failure(
    isomeric_smiles: str,
    config: ConformerEmbeddingConfig,
    status: ConformerTimingStatus,
    error_message: str,
    *,
    n_atoms: int = -1,
    nonisomeric_smiles: str | None = None,
    embed_seconds: float = 0.0,
    mmff_seconds: float = 0.0,
) -> EmbedResult:
    return EmbedResult(
        nonisomeric_smiles=nonisomeric_smiles,
        positions=None,
        atomic_numbers=None,
        timing=ConformerTimingRecord(
            isomeric_smiles=isomeric_smiles,
            n_atoms=n_atoms,
            n_confs_requested=config.n_conformers,
            n_confs_emitted=0,
            t_embed_s=embed_seconds,
            t_mmff_s=mmff_seconds,
            status=status,
            error_msg=error_message,
        ),
    )


def embed_one_smiles(
    isomeric_smiles: str, config: ConformerEmbeddingConfig
) -> EmbedResult:
    """Embed and MMFF-relax one molecule. **Never raises.**

    That includes the exceptions RDKit throws from inside its own C++
    (``RuntimeError: Invariant Violation`` out of the BFGS line search is the
    one seen in production).
    """
    try:
        parsed = Chem.MolFromSmiles(isomeric_smiles)
        if parsed is None:
            return _failure(
                isomeric_smiles, config, "value_error", f"Bad SMILES: {isomeric_smiles}"
            )
        nonisomeric = Chem.MolToSmiles(parsed, isomericSmiles=False)
        mol = Chem.AddHs(parsed)
    except (RuntimeError, ValueError) as error:
        return _failure(
            isomeric_smiles,
            config,
            "value_error",
            f"RDKit raised parsing {isomeric_smiles}: {error}",
        )
    n_atoms = mol.GetNumAtoms()

    params = AllChem.ETKDGv3()
    params.numThreads = 1
    params.maxIterations = config.max_embed_attempts
    params.useRandomCoords = False
    params.useSmallRingTorsions = True
    params.useExpTorsionAnglePrefs = True
    params.enforceChirality = True

    start = perf_counter()
    used_random_coordinates = False
    try:
        conformer_ids = AllChem.EmbedMultipleConfs(
            mol, numConfs=config.n_conformers, params=params
        )
        if not conformer_ids:
            # Bridged ring systems, quaternary centres and macrocycles routinely
            # defeat the distance-geometry start; random initial coordinates are
            # RDKit's documented fallback and recover most of them.
            params.useRandomCoords = True
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol, numConfs=config.n_conformers, params=params
            )
            used_random_coordinates = True
    except (RuntimeError, ValueError) as error:
        return _failure(
            isomeric_smiles,
            config,
            "embed_failed",
            f"RDKit raised during embedding of {isomeric_smiles}: {error}",
            n_atoms=n_atoms,
            nonisomeric_smiles=nonisomeric,
            embed_seconds=perf_counter() - start,
        )
    embed_seconds = perf_counter() - start
    if not conformer_ids:
        return _failure(
            isomeric_smiles,
            config,
            "embed_failed",
            f"No conformers embedded for {isomeric_smiles}",
            n_atoms=n_atoms,
            nonisomeric_smiles=nonisomeric,
            embed_seconds=embed_seconds,
        )

    start = perf_counter()
    try:
        AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            maxIters=config.max_mmff_steps,
            nonBondedThresh=config.mmff_non_bonded_threshold,
            numThreads=1,
        )
    except (RuntimeError, ValueError) as error:
        # Observed in production as ``RuntimeError: Invariant Violation / bad
        # direction in linearSearch ... BFGSOpt.h``.
        return _failure(
            isomeric_smiles,
            config,
            "mmff_failed",
            f"RDKit raised during MMFF relaxation of {isomeric_smiles}: {error}",
            n_atoms=n_atoms,
            nonisomeric_smiles=nonisomeric,
            embed_seconds=embed_seconds,
            mmff_seconds=perf_counter() - start,
        )
    mmff_seconds = perf_counter() - start

    conformers = mol.GetConformers()
    if len(conformers) == 0:
        return _failure(
            isomeric_smiles,
            config,
            "mmff_failed",
            f"No conformers after MMFF for {isomeric_smiles}",
            n_atoms=n_atoms,
            nonisomeric_smiles=nonisomeric,
            embed_seconds=embed_seconds,
            mmff_seconds=mmff_seconds,
        )
    positions = np.stack(
        [
            conformer.GetPositions().astype(np.float64, copy=False)
            for conformer in conformers
        ],
        axis=0,
    )
    return EmbedResult(
        nonisomeric_smiles=nonisomeric,
        positions=positions,
        atomic_numbers=np.array(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int64
        ),
        timing=ConformerTimingRecord(
            isomeric_smiles=isomeric_smiles,
            n_atoms=n_atoms,
            n_confs_requested=config.n_conformers,
            n_confs_emitted=positions.shape[0],
            t_embed_s=embed_seconds,
            t_mmff_s=mmff_seconds,
            status="ok",
            used_random_coordinates=used_random_coordinates,
        ),
    )


def embed_many[Key: Hashable](
    smiles_by_key: Mapping[Key, str],
    config: ConformerEmbeddingConfig,
    *,
    n_workers: int | None = None,
) -> Iterator[tuple[Key, EmbedResult]]:
    """Yield ``(key, EmbedResult)`` for every molecule, in completion order.

    ``n_workers=None`` uses every CPU; ``1`` runs in-process with no pool, which
    keeps a debug run (and tests that monkeypatch :func:`embed_one_smiles`) out
    of the pickling boundary.

    A worker that dies breaks its whole pool and fails every molecule still in
    flight. Those molecules are split in two halves and each half is run in a
    fresh pool, recursively, so the molecule that kills its worker is found in
    a logarithmic number of pools; once it is alone and still kills its
    worker, it is yielded as a ``worker_crashed`` failure. Every key is yielded
    exactly once.
    """
    worker_count = n_workers or os.cpu_count() or 1
    if worker_count == 1:
        for key, isomeric_smiles in smiles_by_key.items():
            yield key, embed_one_smiles(isomeric_smiles, config)
        return

    batches: list[dict[Key, str]] = [dict(smiles_by_key)]
    while batches:
        batch = batches.pop()
        unfinished: dict[Key, str] = {}
        for key, result in _run_pool(batch, config, min(worker_count, len(batch))):
            if result is None:
                unfinished[key] = batch[key]
            else:
                yield key, result
        if not unfinished:
            continue
        if len(unfinished) == 1:
            ((key, isomeric_smiles),) = unfinished.items()
            logger.warning(
                "%s kills its conformer worker; recorded as worker_crashed",
                isomeric_smiles,
            )
            yield (
                key,
                _failure(
                    isomeric_smiles,
                    config,
                    "worker_crashed",
                    "the worker process died while embedding this molecule",
                ),
            )
            continue
        logger.warning(
            "a conformer worker died; retrying %d unfinished molecule(s)",
            len(unfinished),
        )
        keys = list(unfinished)
        middle = len(keys) // 2
        batches.append({key: unfinished[key] for key in keys[middle:]})
        batches.append({key: unfinished[key] for key in keys[:middle]})


def _run_pool[Key: Hashable](
    smiles_by_key: Mapping[Key, str],
    config: ConformerEmbeddingConfig,
    worker_count: int,
) -> Iterator[tuple[Key, EmbedResult | None]]:
    """One pool over ``smiles_by_key``; ``None`` marks a molecule lost to a crash."""
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures: dict[Future[EmbedResult], Key] = {
            executor.submit(embed_one_smiles, isomeric_smiles, config): key
            for key, isomeric_smiles in smiles_by_key.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                yield key, future.result()
            except BrokenProcessPool:
                yield key, None
            except Exception as error:
                # Pickling or another error outside embed_one_smiles itself.
                yield (
                    key,
                    _failure(smiles_by_key[key], config, "other_error", repr(error)),
                )


def charge_and_multiplicity(isomeric_smiles: str) -> tuple[float, float]:
    """``(total formal charge, spin multiplicity)`` of the molecule a SMILES describes.

    Multiplicity is ``2S + 1`` with ``2S`` the number of unpaired (radical)
    electrons, so a closed-shell molecule is 1.0.

    Raises:
        ValueError: if RDKit cannot parse the SMILES.
    """
    molecule = Chem.MolFromSmiles(isomeric_smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES {isomeric_smiles!r}")
    unpaired = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    return float(Chem.GetFormalCharge(molecule)), float(unpaired + 1)
