"""Stream the Chiro small-enantiomer docking dataset into the build pipeline.

The dataset (Adams, Pattanaik & Coley, *Learning 3D Representations of Molecular
Chirality*, arXiv:2110.04383) ships three pandas ``.pkl`` DataFrames (train /
validation / test). Each row is one **3D conformer** of one enantiomer, stored
as an RDKit ``Mol`` with a single embedded conformer plus molecule-level docking
scores. The columns consumed here are::

    ID                          isomeric (stereo) SMILES -- identifies the enantiomer
    SMILES_nostereo             2D SMILES, stereo removed -- shared by the pair
    rdkit_mol_cistrans_stereo   RDKit Mol carrying one 3D conformer
    top_score                   best (lowest) docking score, constant per ID

Each constitution (``SMILES_nostereo``) has exactly two enantiomers; the task is
to rank which enantiomer docks better (lower ``top_score``). That pairing is
materialized for free by the build pipeline: with ``contains_smiles=True`` the
orchestrator derives ``molecule_id`` from the **non**-isomeric SMILES (so the two
enantiomers share it) and ``stereoisomer_id`` from the isomeric SMILES (so all
conformers of one enantiomer share it). The pairwise evaluator groups on those.

Two source quirks are handled here:

* **No explicit hydrogens.** The stored Mols are heavy-atom only, so
  :func:`Chem.AddHs` (``addCoords=True``) re-places hydrogens from the heavy-atom
  geometry before we collapse to ``ase.Atoms`` -- MACE-OFF needs a complete
  molecule. The conformer is already 3D; no embedding is run (the QM9 ingest
  path: ``FilterAtomsStage`` + ``CopyDataStage``).
* **Chiral tags are present** on the Mols, but collapsing to ``Atoms`` (symbols +
  positions) drops them: the model sees coordinates only, which is the honest
  chirality-from-geometry benchmark the paper intends. The isomeric SMILES is
  used solely to assign the integer ``stereoisomer_id`` and is never fed forward.

The literature train/valid/test split is supplied as separate files, so it is
written verbatim into the zarr ``split`` column. ``max_conformers_per_stereoisomer``
caps how many of a stereoisomer's conformers are ingested (first-N in file
order) to bound dataset size.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from ase import Atoms
from rdkit import Chem

from remedi.data_handling.dataset.tasks import (
    Split,
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from remedi.data_handling.dataset_creation.build_stats import LoadStats
from remedi.data_handling.dataset_creation.generators.molecule_generator import (
    MoleculeGenerator,
)
from remedi.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    RegressionData,
    SmilesData,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID

DOCKING_TASK_NAME = "docking_top_score"


def chiro_docking_task_set() -> TaskSet:
    """System-scope single-column regression TaskSet (the docking ``top_score``)."""
    return TaskSet.from_list(
        [
            TaskConfig(
                name=DOCKING_TASK_NAME,
                task_type=TaskType.regression,
                scope=TaskScope.system,
            )
        ]
    )


class ChiroDockingGenerator(MoleculeGenerator):
    """Stream Chiro docking conformers + ``top_score`` from the split pkls.

    ``split_files`` maps each :class:`Split` to its source DataFrame pickle. Rows
    are emitted in train -> valid -> test order; the per-row split code rides in
    ``RegressionData.split`` so the orchestrator materializes it into the zarr.
    """

    def __init__(
        self,
        split_files: dict[Split, Path],
        loading_batch_size: int = 512,
        max_conformers_per_stereoisomer: int | None = 2,
        id_column: str = "ID",
        nonstereo_column: str = "SMILES_nostereo",
        mol_column: str = "rdkit_mol_cistrans_stereo",
        score_column: str = "top_score",
    ):
        self.split_files = {k: Path(v) for k, v in split_files.items()}
        self.loading_batch_size = int(loading_batch_size)
        self.max_conformers_per_stereoisomer = max_conformers_per_stereoisomer
        self.id_column = id_column
        self.nonstereo_column = nonstereo_column
        self.mol_column = mol_column
        self.score_column = score_column
        self.load_stats = LoadStats()

    def task_set(self) -> TaskSet:
        return chiro_docking_task_set()

    def _canonical_smiles(self, isomeric_raw: str) -> SmilesData | None:
        """Canonicalize the stereo SMILES, or ``None`` if RDKit rejects it.

        ``isomeric_smiles`` distinguishes the two enantiomers (its id becomes the
        ``stereoisomer_id``); ``nonisomeric_smiles`` drops stereochemistry and is
        the constitution key shared by the pair (its id becomes ``molecule_id``).
        """
        mol = Chem.MolFromSmiles(isomeric_raw)
        if mol is None:
            return None
        iso = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return SmilesData(
            isomeric_smiles=iso,
            nonisomeric_smiles=Chem.CanonSmiles(iso, useChiral=False),
        )

    @staticmethod
    def _mol_to_atoms(mol: Chem.Mol, smiles: str) -> Atoms | None:
        """Add explicit Hs from the heavy-atom geometry and collapse to Atoms.

        Returns ``None`` if the Mol has no embedded conformer (the source claims
        one per row, but stay defensive). The chiral tag on the Mol is discarded
        by construction -- only symbols and coordinates cross into ``Atoms``.
        """
        if mol.GetNumConformers() == 0:
            return None
        mol_h = Chem.AddHs(mol, addCoords=True)
        conf = mol_h.GetConformer()
        positions = conf.GetPositions()
        symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
        return Atoms(symbols=symbols, positions=positions, info={"smiles": smiles})

    def _iter_kept_rows(self):
        """Yield ``(split_code, atoms, smiles_data, top_score)`` for kept rows.

        Reads each split file, caps conformers per stereoisomer (first-N in file
        order), drops rows with an unparseable SMILES / missing conformer / NaN
        score, and accounts every decision in ``load_stats``.
        """
        import pandas as pd

        cap = self.max_conformers_per_stereoisomer
        for split_code, path in self.split_files.items():
            df = pd.read_pickle(path)
            per_isomer_count: dict[str, int] = {}
            for _, row in df.iterrows():
                self.load_stats.n_raw_rows += 1

                smiles_data = self._canonical_smiles(str(row[self.id_column]))
                if smiles_data is None:
                    self.load_stats.n_invalid_smiles += 1
                    continue

                # Cap conformers per stereoisomer (keyed on the canonical
                # isomeric SMILES so the two enantiomers are capped separately).
                if cap is not None:
                    seen = per_isomer_count.get(smiles_data.isomeric_smiles, 0)
                    if seen >= cap:
                        self.load_stats.n_filtered_out += 1
                        continue
                    per_isomer_count[smiles_data.isomeric_smiles] = seen + 1

                score = row[self.score_column]
                if score is None or (isinstance(score, float) and np.isnan(score)):
                    self.load_stats.n_invalid_smiles += 1
                    continue

                atoms = self._mol_to_atoms(
                    row[self.mol_column], smiles_data.isomeric_smiles
                )
                if atoms is None:
                    self.load_stats.n_invalid_smiles += 1
                    continue

                self.load_stats.n_kept += 1
                yield int(split_code.value), atoms, smiles_data, float(score)

    def __iter__(self) -> Iterator[InputBatch]:
        bs = self.loading_batch_size
        atoms_buf: list[Atoms] = []
        smiles_buf: list[SmilesData] = []
        split_buf: list[int] = []
        score_buf: list[float] = []

        def flush(start_id: int) -> InputBatch:
            n = len(atoms_buf)
            targets = np.asarray(score_buf, dtype=np.float64).reshape(-1, 1)
            masks = np.ones_like(targets, dtype=np.uint8)
            # contains_smiles=True -> the orchestrator derives molecule_id /
            # stereoisomer_id from the SmilesData, so these placeholder ids only
            # have to be unique per row.
            structure_ids = [
                StructureID(structure_id=j, molecule_id=j, stereoisomer_id=j)
                for j in range(start_id, start_id + n)
            ]
            return InputBatch(
                molecules=list(atoms_buf),
                smiles=list(smiles_buf),
                structure_ids=structure_ids,
                # Neutral, closed-shell organics (matches the QM9 / chiral_cat
                # convention): explicit charge 0 / multiplicity 1 so CopyDataStage
                # does not fall back to its unphysical 0.0 multiplicity default.
                total_charge=[0.0] * n,
                multiplicity=[1.0] * n,
                regression_data=RegressionData(
                    targets_system=targets,
                    mask_system=masks,
                    split=np.asarray(split_buf, dtype="u1"),
                ),
            )

        emitted = 0
        for split_code, atoms, smiles_data, score in self._iter_kept_rows():
            atoms_buf.append(atoms)
            smiles_buf.append(smiles_data)
            split_buf.append(split_code)
            score_buf.append(score)
            if len(atoms_buf) >= bs:
                yield flush(emitted)
                emitted += len(atoms_buf)
                atoms_buf, smiles_buf, split_buf, score_buf = [], [], [], []

        if atoms_buf:
            yield flush(emitted)


__all__ = ["DOCKING_TASK_NAME", "ChiroDockingGenerator", "chiro_docking_task_set"]
