"""Tests for the post-refactor dataset creation pipeline.

Covers:
  * XYZMoleculeGenerator reads charge/spin from configurable atoms.info keys
    (and defaults to 0.0 when no key is provided / value is missing).
  * CopyDataStage propagates per-system charge/spin into the DataBatch.
  * MoleculeDataset writes/reads total_charge & total_spin alongside the
    rest of the per-structure data, and TrainingMoleculeDataset exposes
    those values per Sample via atoms_getitem.
  * yield_molecules_collate_fn batches per-system charge/spin scalars.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write as ase_write

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    atoms_getitem,
)
from threedscriptors.data_handling.dataset_creation.generators.xyz_generator import (
    XYZMoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.loading_batch import InputBatch
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
)
from threedscriptors.data_handling.dataset_creation.structure_ids import StructureID
from threedscriptors.data_handling.sample import Sample, yield_molecules_collate_fn


def _make_xyz_file(path: Path, atoms_list: list[Atoms]) -> None:
    ase_write(str(path), atoms_list, format="extxyz")


def test_xyz_generator_reads_charge_and_spin_from_info(tmp_path: Path) -> None:
    a1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a1.info["q"] = 1
    a1.info["S"] = 2

    a2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a2.info["q"] = -1
    a2.info["S"] = 0

    xyz_path = tmp_path / "mols.xyz"
    _make_xyz_file(xyz_path, [a1, a2])

    gen = XYZMoleculeGenerator(
        xyz_file=xyz_path,
        loading_batch_size=10,
        charge_key="q",
        spin_key="S",
    )
    batches = list(gen)
    assert len(batches) == 1
    batch = batches[0]

    assert batch.total_charge == [1.0, -1.0]
    assert batch.total_spin == [2.0, 0.0]


def test_xyz_generator_defaults_to_zero_without_keys(tmp_path: Path) -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    xyz_path = tmp_path / "mols.xyz"
    _make_xyz_file(xyz_path, [atoms])

    gen = XYZMoleculeGenerator(xyz_file=xyz_path, loading_batch_size=10)

    (batch,) = list(gen)
    assert batch.total_charge == [0.0]
    assert batch.total_spin == [0.0]


def test_copy_data_stage_propagates_charge_and_spin() -> None:
    atoms_a = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms_b = Atoms("CH4", positions=np.zeros((5, 3)))

    input_batch = InputBatch(
        smiles=None,
        molecules=[atoms_a, atoms_b],
        structure_ids=[
            StructureID(structure_id=0, molecule_id=0, stereoisomer_id=0),
            StructureID(structure_id=1, molecule_id=1, stereoisomer_id=1),
        ],
        total_charge=[1.0, -2.0],
        total_spin=[0.5, 1.5],
    )

    stage = CopyDataStage(dtype=torch.float64)
    _, data_batch = stage(input_batch, None)

    assert data_batch.total_charge.dtype == torch.float64
    assert data_batch.total_spin.dtype == torch.float64
    torch.testing.assert_close(
        data_batch.total_charge, torch.tensor([1.0, -2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        data_batch.total_spin, torch.tensor([0.5, 1.5], dtype=torch.float64)
    )


def test_copy_data_stage_defaults_when_input_missing() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    input_batch = InputBatch(
        smiles=None,
        molecules=[atoms],
        structure_ids=[StructureID(structure_id=0, molecule_id=0, stereoisomer_id=0)],
    )
    _, data_batch = CopyDataStage(dtype=torch.float64)(input_batch, None)
    torch.testing.assert_close(
        data_batch.total_charge, torch.zeros(1, dtype=torch.float64)
    )
    torch.testing.assert_close(
        data_batch.total_spin, torch.zeros(1, dtype=torch.float64)
    )


def test_orchestrator_round_trip_persists_charge_and_spin(tmp_path: Path) -> None:
    a1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a1.info["q"] = 1
    a1.info["S"] = 2

    a2 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    a2.info["q"] = -1
    a2.info["S"] = 1

    xyz_path = tmp_path / "src.xyz"
    _make_xyz_file(xyz_path, [a1, a2])

    out_dir = tmp_path / "zarr_out"

    orchestrator = DatasetConstructionOrchestrator(
        pipeline=[CopyDataStage(dtype=torch.float64)],
        batch_generator=XYZMoleculeGenerator(
            xyz_file=xyz_path,
            loading_batch_size=10,
            charge_key="q",
            spin_key="S",
        ),
        construction_config=DatasetCreationConfig(path=out_dir, N_structures=10),
        dataset_config=DatasetConfig(
            atom_chunk=8, molecule_chunk=4, contains_smiles=False
        ),
    )
    orchestrator.build_dataset()

    reopened = MoleculeDataset.open_existing_dataset_from_dir(out_dir)
    assert reopened.N_structures == 2
    assert reopened.N_atoms == 4

    n = reopened.N_structures
    np.testing.assert_allclose(
        np.asarray(reopened.total_charge[:n]), np.array([1.0, -1.0], dtype="f4")
    )
    np.testing.assert_allclose(
        np.asarray(reopened.total_spin[:n]), np.array([2.0, 1.0], dtype="f4")
    )


def test_atoms_getitem_returns_charge_and_spin_per_sample(tmp_path: Path) -> None:
    a1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    a1.info["q"] = 1
    a1.info["S"] = 2
    a2 = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    a2.info["q"] = 0
    a2.info["S"] = 1

    xyz_path = tmp_path / "src.xyz"
    _make_xyz_file(xyz_path, [a1, a2])
    out_dir = tmp_path / "zarr_out"

    DatasetConstructionOrchestrator(
        pipeline=[CopyDataStage(dtype=torch.float64)],
        batch_generator=XYZMoleculeGenerator(
            xyz_file=xyz_path,
            loading_batch_size=10,
            charge_key="q",
            spin_key="S",
        ),
        construction_config=DatasetCreationConfig(path=out_dir, N_structures=10),
        dataset_config=DatasetConfig(
            atom_chunk=8, molecule_chunk=4, contains_smiles=False
        ),
    ).build_dataset()

    train_ds = TrainingMoleculeDataset(out_dir, atoms_getitem, in_memory=True)
    assert len(train_ds) == 2

    sample0 = train_ds[0]
    sample1 = train_ds[1]

    assert sample0.atomic_positions.shape == (2, 3)
    assert sample1.atomic_positions.shape == (2, 3)

    torch.testing.assert_close(sample0.total_charge, torch.tensor(1.0))
    torch.testing.assert_close(sample0.total_spin, torch.tensor(2.0))
    torch.testing.assert_close(sample1.total_charge, torch.tensor(0.0))
    torch.testing.assert_close(sample1.total_spin, torch.tensor(1.0))


def test_yield_molecules_collate_batches_charge_and_spin() -> None:
    s1 = Sample(
        atomic_positions=torch.zeros((2, 3)),
        atomic_numbers=torch.tensor([1, 1], dtype=torch.uint8),
        total_charge=torch.tensor(1.0),
        total_spin=torch.tensor(2.0),
    )
    s2 = Sample(
        atomic_positions=torch.zeros((3, 3)),
        atomic_numbers=torch.tensor([6, 1, 1], dtype=torch.uint8),
        total_charge=torch.tensor(-1.0),
        total_spin=torch.tensor(0.5),
    )

    batched = yield_molecules_collate_fn([s1, s2])

    assert batched.atomic_positions.shape == (5, 3)
    assert batched.atomic_numbers.shape == (5,)
    torch.testing.assert_close(
        batched.system_index, torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    )
    torch.testing.assert_close(batched.total_charge, torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(batched.total_spin, torch.tensor([2.0, 0.5]))


def test_yield_molecules_collate_requires_charge_and_spin() -> None:
    s = Sample(
        atomic_positions=torch.zeros((1, 3)),
        atomic_numbers=torch.tensor([1], dtype=torch.uint8),
    )
    with pytest.raises((TypeError, ValueError, AttributeError)):
        yield_molecules_collate_fn([s])
