"""Tests for the ChiralCat generator + the stratified group split.

Covers:
  * ChiralCatGenerator reads the integer ``label`` and ``smiles`` from the
    extxyz comment line, emits one multiclass column and per-row split codes.
  * The single task is TaskType.multiclass in the task set.
  * stratified_group_split keeps groups (non-isomeric SMILES) whole and spreads
    every class across train/valid/test.
  * A full orchestrator round-trip persists the class labels + split column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import ElementSet, Split, TaskType
from remedi.data_handling.dataset_creation.generators.chiral_cat_generator import (
    CHIRAL_TASK_NAME,
    ChiralCatGenerator,
    chiral_cat_task_set,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)
from remedi.data_handling.dataset_creation.splits import (
    stratified_group_split,
)

# (smiles, label) — five distinct-graph molecules per class over three classes,
# enough groups that the stratified split populates all of train/valid/test even
# at small N (mirrors the real dataset where every class has >=37 molecules).
# Labels are arbitrary here; only the per-class grouping matters for the split.
_FRAMES = [
    ("C", 0),
    ("CC", 0),
    ("CCC", 0),
    ("CCCC", 0),
    ("CCCCC", 0),
    ("CCO", 1),
    ("CCN", 1),
    ("CCCO", 1),
    ("CCCN", 1),
    ("CCCCO", 1),
    ("c1ccccc1", 2),
    ("Cc1ccccc1", 2),
    ("CCc1ccccc1", 2),
    ("c1ccncc1", 2),
    ("Cc1ccncc1", 2),
]


def _make_chiral_xyz(path: Path) -> None:
    frames = []
    for i, (smi, label) in enumerate(_FRAMES):
        a = Atoms(
            "CH4",
            positions=np.zeros((5, 3)),
            info={"smiles": smi, "label": label, "index": i},
        )
        frames.append(a)
    ase_write(str(path), frames, format="extxyz")


def test_task_set_is_single_multiclass_column() -> None:
    ts = chiral_cat_task_set()
    assert [c.name for c in ts.system_cols] == [CHIRAL_TASK_NAME]
    assert ts.system_cols[0].task_type == TaskType.multiclass
    assert not ts.atom_cols


def test_generator_reads_labels_and_split(tmp_path: Path) -> None:
    xyz = tmp_path / "chiral.extxyz"
    _make_chiral_xyz(xyz)

    gen = ChiralCatGenerator(
        xyz_file=xyz, loading_batch_size=100, train_frac=0.6, val_frac=0.2
    )
    (batch,) = list(gen)

    assert batch.regression_data is not None
    labels = batch.regression_data.targets_system[:, 0]
    np.testing.assert_array_equal(labels, [lbl for _, lbl in _FRAMES])
    assert batch.regression_data.targets_system.shape == (len(_FRAMES), 1)
    assert (batch.regression_data.mask_system == 1).all()
    assert gen.load_stats.n_kept == len(_FRAMES)
    # All three partitions are represented in the materialized split codes.
    codes = set(batch.regression_data.split.tolist())
    assert Split.train.value in codes
    assert Split.valid.value in codes
    assert Split.test.value in codes
    # Neutral, closed-shell convention.
    assert batch.total_charge == [0.0] * len(_FRAMES)
    assert batch.multiplicity == [1.0] * len(_FRAMES)


def test_generator_drops_invalid_smiles(tmp_path: Path) -> None:
    good = Atoms("CH4", positions=np.zeros((5, 3)), info={"smiles": "C", "label": 0})
    bad = Atoms("CH4", positions=np.zeros((5, 3)), info={"smiles": "xyz!", "label": 1})
    xyz = tmp_path / "chiral.extxyz"
    ase_write(str(xyz), [good, bad], format="extxyz")

    gen = ChiralCatGenerator(xyz_file=xyz, loading_batch_size=100)
    (batch,) = list(gen)
    assert len(batch) == 1
    assert gen.load_stats.n_raw_rows == 2
    assert gen.load_stats.n_invalid_smiles == 1


def test_stratified_group_split_keeps_groups_whole_and_spreads_classes() -> None:
    # 30 groups per class over 3 classes; one row per group.
    labels = np.repeat([0, 1, 2], 30)
    groups = np.array([f"g{i}" for i in range(len(labels))], dtype=object)
    tr, va, te = stratified_group_split(labels, groups, 0.6, 0.2, seed=0)

    # Partition every row exactly once, no overlap.
    assert sorted([*tr, *va, *te]) == list(range(len(labels)))
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    # Every class appears in every partition.
    for part in (tr, va, te):
        assert set(labels[part].tolist()) == {0, 1, 2}


def test_stratified_group_split_no_group_leakage() -> None:
    # Two rows per group; the pair must land in the same partition.
    labels = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])
    groups = np.array(
        ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"], dtype=object
    )
    tr, va, te = stratified_group_split(labels, groups, 0.5, 0.25, seed=1)
    row_to_part = {}
    for name, part in (("tr", tr), ("va", va), ("te", te)):
        for r in part:
            row_to_part[r] = name
    for g in set(groups):
        rows = [i for i, gg in enumerate(groups) if gg == g]
        assert len({row_to_part[r] for r in rows}) == 1, f"group {g} leaked"


def test_orchestrator_round_trip_persists_labels_and_split(tmp_path: Path) -> None:
    xyz = tmp_path / "chiral.extxyz"
    _make_chiral_xyz(xyz)
    out_dir = tmp_path / "zarr_out"

    gen = ChiralCatGenerator(
        xyz_file=xyz, loading_batch_size=100, train_frac=0.6, val_frac=0.2
    )
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
    assert ds.N_structures == len(_FRAMES)
    assert ds.config.tasks is not None
    assert ds.config.tasks.system_cols[0].task_type == TaskType.multiclass
    labels = np.asarray(ds.targets_system[:, 0])
    np.testing.assert_array_equal(labels, [lbl for _, lbl in _FRAMES])
    codes = set(np.asarray(ds.split[:]).tolist())
    assert {Split.train.value, Split.valid.value, Split.test.value} <= codes
