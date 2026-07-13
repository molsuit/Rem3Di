"""Tests for the PhysicochemicalDescriptorStage (loaded-structure path)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    PhysicochemicalDescriptorStageConfig,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset_creation.loading_batch import (
    InputBatch,
    SmilesData,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    PhysicochemicalDescriptorStage,
)
from remedi.data_handling.dataset_creation.structure_ids import StructureID


def _atoms(smiles: str) -> Atoms:
    """A loaded-style ase.Atoms (3D, with H) — atom order from the SDF-like mol."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    return Atoms(
        numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
    )


def _batch(smiles_list, smiles_field=None):
    smiles_field = smiles_list if smiles_field is None else smiles_field
    return InputBatch(
        smiles=[
            SmilesData(nonisomeric_smiles=s or "", isomeric_smiles=s or "")
            for s in smiles_field
        ],
        molecules=[_atoms(s) for s in smiles_list],
        structure_ids=[
            StructureID(structure_id=i, molecule_id=i, stereoisomer_id=i)
            for i in range(len(smiles_list))
        ],
    )


def test_to_task_set_column_order_matches_names():
    cfg = PhysicochemicalDescriptorStageConfig()
    ts = cfg.to_task_set()
    assert [c.name for c in ts.system_cols] == cfg.descriptor_names
    assert all(c.scope.value == "system" for c in ts.system_cols)


def test_stage_fills_targets_and_masks():
    cfg = PhysicochemicalDescriptorStageConfig()
    stage = PhysicochemicalDescriptorStage(cfg, num_workers=1)
    ib = _batch(["CCO", "c1ccccc1", "CC(=O)O"])
    ib2, _ = stage(ib, None)
    rd = ib2.regression_data
    p = len(cfg.descriptor_names)
    assert rd.targets_system.shape == (3, p)
    assert rd.targets_system.dtype == np.float32
    assert rd.mask_system.shape == (3, p)
    assert rd.mask_system.all()  # all valid for clean organic mols


def test_stage_masks_missing_smiles_but_keeps_3d():
    cfg = PhysicochemicalDescriptorStageConfig()
    stage = PhysicochemicalDescriptorStage(cfg, num_workers=1)
    # Second structure has no SMILES; its geometry is still valid.
    ib = _batch(["CCO", "c1ccccc1"], smiles_field=["CCO", ""])
    ib2, _ = stage(ib, None)
    mask = ib2.regression_data.mask_system
    names = cfg.descriptor_names
    for i, n in enumerate(names):
        is_3d = n in ("sasa", "polar_sasa_fraction")
        if is_3d:
            assert mask[1, i] == 1
        else:
            assert mask[1, i] == 0
    # Masked entries are zeroed (never NaN) so a forgotten mask can't propagate.
    assert np.all(np.isfinite(ib2.regression_data.targets_system))


class _SingleBatchGen:
    """Minimal generator yielding one pre-built InputBatch (for build tests)."""

    def __init__(self, batch: InputBatch):
        self._batch = batch

    def __iter__(self):
        yield self._batch


def test_stage_build_round_trip_writes_targets_system(tmp_path: Path):
    cfg = PhysicochemicalDescriptorStageConfig()
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    batch = _batch(smiles)
    out_dir = tmp_path / "zarr_out"

    DatasetConstructionOrchestrator(
        pipeline=[
            PhysicochemicalDescriptorStage(cfg, num_workers=1),
            CopyDataStage(dtype=torch.float64),
        ],
        batch_generator=_SingleBatchGen(batch),
        construction_config=DatasetCreationConfig(path=out_dir, N_structures=10),
        dataset_config=DatasetConfig(
            atom_chunk=8,
            molecule_chunk=4,
            contains_smiles=True,
            tasks=cfg.to_task_set(),
        ),
    ).build_dataset()

    reopened = MoleculeDataset.open_existing_dataset_from_dir(out_dir)
    assert reopened.N_structures == 3

    # TaskSet round-trips through dataset_config.yaml in the right column order.
    names = [c.name for c in reopened.config.tasks.system_cols]
    assert names == cfg.descriptor_names

    targets = np.asarray(reopened.targets_system[:])
    masks = np.asarray(reopened.mask_system[:])
    assert targets.shape == (3, len(cfg.descriptor_names))
    assert masks.shape == (3, len(cfg.descriptor_names))
    assert masks.all()
    assert np.all(np.isfinite(targets))

    # n_heavy_atoms column matches the ground truth (ethanol=3, benzene=6, acetic=4).
    col = reopened.config.tasks.system_map["n_heavy_atoms"]
    np.testing.assert_allclose(targets[:, col], [3.0, 6.0, 4.0])
    reopened.close()
