"""Stream the curated/recalculated QM9 extxyz into the build pipeline.

The curated QM9 file stores every molecule as one extxyz frame whose comment
line carries the 15 GDB-9 properties plus the reference ``SMILES`` as ``key=value``
pairs, e.g.::

    Properties=species:S:1:pos:R:3 index=1 A=157.71 ... Cv=6.469 SMILES="C"

``ase.io.iread`` lifts each of those into ``atoms.info``, so this generator just
reads the requested property columns straight out of ``atoms.info`` (no separate
label file like tmQM) and emits the SMILES as canonical ``SmilesData`` so the
molecule_id / stereoisomer_id bookkeeping matches every other SMILES-bearing
source. Atom-count / element gates live in ``FilterAtomsStage``; this generator's
only jobs are parsing properties and canonicalizing SMILES.
"""

from enum import StrEnum
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import iread
from rdkit import Chem

from threedscriptors.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from threedscriptors.data_handling.dataset_creation.build_stats import LoadStats
from threedscriptors.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
    SmilesData,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID


class QM9Property(StrEnum):
    """A QM9 regression target; the value is its key in the extxyz comment line.

    Units follow the GDB-9 reference: rotational constants A/B/C [GHz], dipole
    moment mu [D], isotropic polarizability alpha [a0^3], orbital energies
    homo/lumo and gap [Ha], <R^2> r2 [a0^2], zero-point vibrational energy zpve
    [Ha], internal energies U0/U and enthalpy H and Gibbs free energy G [Ha],
    and heat capacity Cv [cal/mol/K].
    """

    A = "A"
    B = "B"
    C = "C"
    mu = "mu"
    alpha = "alpha"
    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    r2 = "r2"
    zpve = "zpve"
    U0 = "U0"
    U = "U"
    H = "H"
    G = "G"
    Cv = "Cv"


ALL_QM9_TASKS: list[QM9Property] = list(QM9Property)


def qm9_task_set(tasks: list[QM9Property]) -> TaskSet:
    """Build a system-scope regression TaskSet, one column per QM9 property.

    Column order matches the order of ``tasks``, which the generator uses when
    it stacks ``targets_system``; keep the two in lockstep.
    """
    return TaskSet.from_list(
        [
            TaskConfig(
                name=t.value,
                task_type=TaskType.regression,
                scope=TaskScope.system,
            )
            for t in tasks
        ]
    )


class QM9Generator(MoleculeGenerator):
    """Stream curated-QM9 Atoms + 15 GDB-9 targets from one extxyz file."""

    def __init__(
        self,
        xyz_file: Path,
        tasks: list[QM9Property] = ALL_QM9_TASKS,
        loading_batch_size: int = 10000,
        smiles_key: str = "SMILES",
    ):
        self.xyz_file = xyz_file
        self.tasks = list(tasks)
        self.loading_batch_size = int(loading_batch_size)
        self.smiles_key = smiles_key
        self.load_stats = LoadStats()

    def task_set(self) -> TaskSet:
        """TaskSet describing the columns this generator writes (build-config glue)."""
        return qm9_task_set(self.tasks)

    def _canonical_smiles(self, raw: str) -> SmilesData | None:
        """Canonicalize one reference SMILES, or None if RDKit rejects it.

        Matches the GeomGenerator convention: ``isomeric_smiles`` is the RDKit
        canonical isomeric form and ``nonisomeric_smiles`` drops stereochemistry.
        """
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return None
        iso = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return SmilesData(
            isomeric_smiles=iso,
            nonisomeric_smiles=Chem.CanonSmiles(iso, useChiral=False),
        )

    def _read_targets(self, atoms: Atoms) -> np.ndarray:
        return np.array(
            [float(atoms.info[t.value]) for t in self.tasks], dtype=np.float64
        )

    def _make_batch(
        self,
        batch_atoms: list[Atoms],
        batch_smiles: list[SmilesData],
        batch_ids: list[StructureID],
        batch_targets: list[np.ndarray],
    ) -> InputBatch:
        targets = np.stack(batch_targets)
        masks = np.ones_like(targets, dtype=np.uint8)
        n = len(batch_atoms)
        return InputBatch(
            molecules=batch_atoms,
            smiles=batch_smiles,
            structure_ids=batch_ids,
            # GDB-9 is entirely neutral, closed-shell organics: charge 0 and
            # spin multiplicity (2S+1) = 1. Set both explicitly so CopyDataStage
            # does not fall back to its 0.0 default (multiplicity 0 is unphysical
            # and wrong input for MACE / PolarMACE, which want 1 for a singlet).
            total_charge=[0.0] * n,
            multiplicity=[1.0] * n,
            regression_data=RegressionData(
                targets_system=targets, mask_system=masks
            ),
        )

    def __iter__(self):
        suppl = iread(self.xyz_file, index=":")

        batch_atoms: list[Atoms] = []
        batch_smiles: list[SmilesData] = []
        batch_ids: list[StructureID] = []
        batch_targets: list[np.ndarray] = []

        kept = 0
        for atoms in suppl:
            self.load_stats.n_raw_rows += 1

            raw_smiles = atoms.info.get(self.smiles_key)
            if raw_smiles is None:
                self.load_stats.n_invalid_smiles += 1
                continue
            smiles_data = self._canonical_smiles(str(raw_smiles))
            if smiles_data is None:
                self.load_stats.n_invalid_smiles += 1
                continue

            batch_atoms.append(atoms)
            batch_smiles.append(smiles_data)
            batch_ids.append(
                StructureID(
                    structure_id=kept, molecule_id=kept, stereoisomer_id=kept
                )
            )
            batch_targets.append(self._read_targets(atoms))
            kept += 1
            # Keep the running total live so the build summary is correct even
            # when the orchestrator stops us early on N_structures.
            self.load_stats.n_kept = kept

            if len(batch_atoms) >= self.loading_batch_size:
                yield self._make_batch(
                    batch_atoms, batch_smiles, batch_ids, batch_targets
                )
                batch_atoms, batch_smiles, batch_ids, batch_targets = [], [], [], []

        if batch_atoms:
            yield self._make_batch(
                batch_atoms, batch_smiles, batch_ids, batch_targets
            )
