"""Stream the ChiralCat 5-class chirality-type dataset into the build pipeline.

ChiralCat ships one extended-XYZ file (``chiral_structures.extxyz``) with one
3D conformer per molecule, explicit hydrogens, and the per-frame metadata on the
comment line::

    Properties=species:S:1:pos:R:3 index=<i> label=<L> class=<name> smiles="<SMILES>" pbc="F F F"

``ase.io.iread`` lifts those into ``atoms.info`` so this generator reads the
integer ``label`` (the 0-4 chirality class) and the reference ``smiles`` straight
out of ``atoms.info`` -- no separate label file. Because the geometries are
supplied directly (ETKDG/MMFF94 conformers), this follows the QM9 ingest path:
``FilterAtomsStage`` + ``CopyDataStage`` with no conformer generation.

The single classification column is a :class:`TaskType.multiclass` task -- the
integer class index is written verbatim into ``targets_system[:, 0]`` (the eval
derives the class count from the data). The train/valid/test split is computed
once over all kept molecules with :func:`stratified_group_split`: groups are the
non-isomeric SMILES (so enantiomer pairs never straddle a fold) and the split is
class-stratified so the rare ``helical`` / ``planar`` classes appear in every
partition. Class imbalance is summarized in ``DATASET.md``::

    0 achiral 10001 | 1 central 6399 | 2 axial 527 | 3 helical 37 | 4 planar 59
"""

from __future__ import annotations

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
from threedscriptors.data_handling.dataset_creation.splits import (
    split_codes,
    stratified_group_split,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID

CHIRAL_TASK_NAME = "chirality_type"


def chiral_cat_task_set() -> TaskSet:
    """System-scope multi-class TaskSet (one column, the chirality class)."""
    return TaskSet.from_list(
        [
            TaskConfig(
                name=CHIRAL_TASK_NAME,
                task_type=TaskType.multiclass,
                scope=TaskScope.system,
            )
        ]
    )


class ChiralCatGenerator(MoleculeGenerator):
    """Stream ChiralCat Atoms + the 5-class chirality label from one extxyz."""

    def __init__(
        self,
        xyz_file: Path,
        loading_batch_size: int = 4096,
        label_key: str = "label",
        smiles_key: str = "smiles",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        split_seed: int = 0,
    ):
        self.xyz_file = Path(xyz_file)
        self.loading_batch_size = int(loading_batch_size)
        self.label_key = label_key
        self.smiles_key = smiles_key
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.split_seed = split_seed
        self.load_stats = LoadStats()

    def task_set(self) -> TaskSet:
        return chiral_cat_task_set()

    def _canonical_smiles(self, raw: str) -> SmilesData | None:
        """Canonicalize one reference SMILES, or None if RDKit rejects it.

        ``isomeric_smiles`` is the RDKit canonical isomeric form (distinguishes
        enantiomers); ``nonisomeric_smiles`` drops stereochemistry and is the
        grouping key that keeps enantiomer pairs in the same split.
        """
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return None
        iso = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return SmilesData(
            isomeric_smiles=iso,
            nonisomeric_smiles=Chem.CanonSmiles(iso, useChiral=False),
        )

    def _load(
        self,
    ) -> tuple[list[Atoms], list[SmilesData], np.ndarray, np.ndarray]:
        """Read every frame, keeping rows with a valid SMILES + integer label.

        Returns the kept Atoms, their canonical SMILES, the class-label array
        (int) and the per-row split codes (computed once over all kept rows).
        """
        atoms_list: list[Atoms] = []
        smiles_list: list[SmilesData] = []
        labels: list[int] = []

        for atoms in iread(self.xyz_file, index=":"):
            self.load_stats.n_raw_rows += 1
            raw_smiles = atoms.info.get(self.smiles_key)
            if raw_smiles is None:
                self.load_stats.n_invalid_smiles += 1
                continue
            smiles_data = self._canonical_smiles(str(raw_smiles))
            if smiles_data is None:
                self.load_stats.n_invalid_smiles += 1
                continue
            if self.label_key not in atoms.info:
                self.load_stats.n_invalid_smiles += 1
                continue
            atoms_list.append(atoms)
            smiles_list.append(smiles_data)
            labels.append(int(atoms.info[self.label_key]))

        self.load_stats.n_kept = len(atoms_list)
        label_arr = np.asarray(labels, dtype=np.int64)
        groups = np.asarray([sd.nonisomeric_smiles for sd in smiles_list], dtype=object)
        if len(atoms_list) == 0:
            codes = np.empty(0, dtype="u1")
        else:
            train_idx, valid_idx, test_idx = stratified_group_split(
                label_arr,
                groups,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                seed=self.split_seed,
            )
            codes = split_codes(len(atoms_list), train_idx, valid_idx, test_idx)
        return atoms_list, smiles_list, label_arr, codes

    def __iter__(self):
        atoms_list, smiles_list, label_arr, codes = self._load()
        n = len(atoms_list)
        if n == 0:
            return

        bs = self.loading_batch_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            structure_ids = [
                StructureID(structure_id=j, molecule_id=j, stereoisomer_id=j)
                for j in range(start, end)
            ]
            # Single multi-class column: the integer class index is written as
            # a float into targets_system[:, 0]; the eval rounds it back.
            targets = label_arr[start:end].astype(np.float64).reshape(-1, 1)
            masks = np.ones_like(targets, dtype=np.uint8)
            yield InputBatch(
                molecules=atoms_list[start:end],
                smiles=smiles_list[start:end],
                structure_ids=structure_ids,
                # Neutral, closed-shell organics (matches the QM9 convention):
                # explicit charge 0 and spin multiplicity 1 so CopyDataStage
                # does not fall back to its unphysical 0.0 multiplicity default.
                total_charge=[0.0] * (end - start),
                multiplicity=[1.0] * (end - start),
                regression_data=RegressionData(
                    targets_system=targets,
                    mask_system=masks,
                    split=codes[start:end],
                ),
            )
