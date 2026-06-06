"""Tests for the Chiro docking generator + its orchestrator round-trip.

Covers:
  * The single task is a TaskType.regression ``docking_top_score`` column.
  * The generator re-adds explicit hydrogens (heavy-atom-only source -> MACE-OFF
    needs Hs), reads ``top_score`` into the regression column, caps conformers
    per stereoisomer, and stamps the per-file split code.
  * A full orchestrator round-trip persists the score + split column and groups
    enantiomers: the two stereoisomers of a constitution share ``molecule_id``
    but get distinct ``isomer_id``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.tasks import ElementSet, Split, TaskType
from threedscriptors.data_handling.dataset_creation.generators.chiro_docking_generator import (
    DOCKING_TASK_NAME,
    ChiroDockingGenerator,
    chiro_docking_task_set,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

# Two constitutions, each an enantiomer pair, so the round-trip exercises the
# molecule/stereoisomer grouping the pairwise eval relies on.
_PAIRS = [
    ("C[C@H](N)O", "C[C@@H](N)O", "CC(N)O"),
    ("C[C@H](Cl)Br", "C[C@@H](Cl)Br", "CC(Cl)Br"),
]


def _mol3d(smiles: str) -> Chem.Mol:
    """Embed one 3D conformer then strip Hs, mimicking the stored heavy-atom Mol."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    return Chem.RemoveHs(mol)


def _make_docking_pickle(
    path: Path, rows: list[tuple[str, str, float]], confs_per_row: int = 1
) -> None:
    """Write a DataFrame pkl with ``confs_per_row`` conformer rows per (ID)."""
    records = []
    for iso_smiles, nostereo, top_score in rows:
        for _ in range(confs_per_row):
            records.append(
                {
                    "ID": iso_smiles,
                    "SMILES_nostereo": nostereo,
                    "rdkit_mol_cistrans_stereo": _mol3d(iso_smiles),
                    "top_score": top_score,
                }
            )
    pd.DataFrame.from_records(records).to_pickle(path)


def _split_files(tmp_path: Path) -> dict[Split, Path]:
    # Each split gets one full enantiomer pair so all three codes appear.
    train = tmp_path / "train.pkl"
    valid = tmp_path / "valid.pkl"
    test = tmp_path / "test.pkl"
    _make_docking_pickle(
        train,
        [
            (_PAIRS[0][0], _PAIRS[0][2], -5.0),
            (_PAIRS[0][1], _PAIRS[0][2], -4.0),
        ],
    )
    _make_docking_pickle(
        valid,
        [
            (_PAIRS[1][0], _PAIRS[1][2], -6.0),
            (_PAIRS[1][1], _PAIRS[1][2], -5.5),
        ],
    )
    _make_docking_pickle(
        test,
        [
            (_PAIRS[1][0], _PAIRS[1][2], -6.0),
            (_PAIRS[1][1], _PAIRS[1][2], -5.5),
        ],
    )
    return {Split.train: train, Split.valid: valid, Split.test: test}


def test_task_set_is_single_regression_column() -> None:
    ts = chiro_docking_task_set()
    assert [c.name for c in ts.system_cols] == [DOCKING_TASK_NAME]
    assert ts.system_cols[0].task_type == TaskType.regression
    assert not ts.atom_cols


def test_generator_adds_hydrogens_and_reads_scores(tmp_path: Path) -> None:
    files = _split_files(tmp_path)
    gen = ChiroDockingGenerator(split_files=files, loading_batch_size=100)
    batches = list(gen)
    atoms = [a for b in batches for a in b.molecules]
    scores = np.concatenate(
        [b.regression_data.targets_system[:, 0] for b in batches]
    )
    codes = np.concatenate([b.regression_data.split for b in batches])

    assert gen.load_stats.n_kept == 6  # 2 + 2 + 2
    # Heavy-atom Mols are stored without H; the generator re-adds them.
    assert all(any(s == "H" for s in a.get_chemical_symbols()) for a in atoms)
    # Scores land in the regression column; every label is valid.
    assert set(np.round(scores, 1)) == {-5.0, -4.0, -6.0, -5.5}
    assert all((b.regression_data.mask_system == 1).all() for b in batches)
    # Each file stamps its split code.
    assert set(codes.tolist()) == {
        Split.train.value,
        Split.valid.value,
        Split.test.value,
    }


def test_generator_caps_conformers_per_stereoisomer(tmp_path: Path) -> None:
    f = tmp_path / "train.pkl"
    # 4 conformer rows for each of the two enantiomers.
    _make_docking_pickle(
        f,
        [(_PAIRS[0][0], _PAIRS[0][2], -5.0), (_PAIRS[0][1], _PAIRS[0][2], -4.0)],
        confs_per_row=4,
    )
    gen = ChiroDockingGenerator(
        split_files={Split.train: f}, max_conformers_per_stereoisomer=2
    )
    (batch,) = list(gen)
    # 2 enantiomers x cap 2 = 4 kept; the other 4 rows are filtered out.
    assert gen.load_stats.n_raw_rows == 8
    assert gen.load_stats.n_kept == 4
    assert gen.load_stats.n_filtered_out == 4
    assert len(batch) == 4


def test_orchestrator_round_trip_groups_enantiomers(tmp_path: Path) -> None:
    files = _split_files(tmp_path)
    out_dir = tmp_path / "zarr_out"
    gen = ChiroDockingGenerator(split_files=files, loading_batch_size=100)
    DatasetConstructionOrchestrator(
        pipeline=[
            FilterAtomsStage(
                config=FilterAtomsStageConfig(element_set=ElementSet.mace_off)
            ),
            CopyDataStage(dtype=torch.float64),
        ],
        batch_generator=gen,
        construction_config=DatasetCreationConfig(path=out_dir, N_structures=None),
        dataset_config=DatasetConfig(
            atom_chunk=8, molecule_chunk=4, tasks=gen.task_set()
        ),
    ).build_dataset()

    ds = MoleculeDataset.open_existing_dataset_from_dir(out_dir)
    assert ds.N_structures == 6
    assert ds.config.tasks.system_cols[0].task_type == TaskType.regression

    mol_ids = np.asarray(ds.molecule_ids[:])
    iso_ids = np.asarray(ds.isomer_ids[:])
    # 2 constitutions -> 2 distinct molecule_ids; 4 distinct stereoisomers.
    assert len(set(mol_ids.tolist())) == 2
    assert len(set(iso_ids.tolist())) == 4
    # Every constitution carries exactly two stereoisomers (an enantiomer pair).
    for mol in set(mol_ids.tolist()):
        isos = set(iso_ids[mol_ids == mol].tolist())
        assert len(isos) == 2

    codes = set(np.asarray(ds.split[:]).tolist())
    assert {Split.train.value, Split.valid.value, Split.test.value} <= codes
