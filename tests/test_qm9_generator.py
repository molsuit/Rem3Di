"""Tests for the curated-QM9 generator + build pipeline.

Covers:
  * QM9Generator reads the 15 GDB-9 properties straight out of the extxyz
    comment line (atoms.info) into targets_system, in task-set column order.
  * Reference SMILES are canonicalized into SmilesData (iso + non-iso).
  * Rows whose SMILES RDKit rejects are dropped and counted in load_stats.
  * A full orchestrator round-trip persists targets, masks, and SMILES.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.tasks import ElementSet
from threedscriptors.data_handling.dataset_creation.generators.qm9_generator import (
    ALL_QM9_TASKS,
    QM9Generator,
    QM9Property,
    qm9_task_set,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

# Two QM9-style frames: methane and water, with all 15 property keys + SMILES.
_PROP_KEYS = [p.value for p in ALL_QM9_TASKS]
_CH4 = dict(
    zip(_PROP_KEYS, np.arange(15.0), strict=True),
)
_H2O = dict(
    zip(_PROP_KEYS, np.arange(100.0, 115.0), strict=True),
)


def _make_qm9_xyz(path: Path) -> None:
    ch4 = Atoms("CH4", positions=np.zeros((5, 3)), info={**_CH4, "SMILES": "C"})
    h2o = Atoms("H2O", positions=np.zeros((3, 3)), info={**_H2O, "SMILES": "O"})
    ase_write(str(path), [ch4, h2o], format="extxyz")


def test_qm9_task_set_column_order() -> None:
    ts = qm9_task_set(ALL_QM9_TASKS)
    assert [c.name for c in ts.system_cols] == _PROP_KEYS
    assert ts.system_map["A"] == 0
    assert ts.system_map["Cv"] == 14
    assert not ts.atom_cols


def test_generator_reads_all_targets_and_smiles(tmp_path: Path) -> None:
    xyz_path = tmp_path / "qm9.xyz"
    _make_qm9_xyz(xyz_path)

    gen = QM9Generator(xyz_file=xyz_path, loading_batch_size=10)
    (batch,) = list(gen)

    assert batch.regression_data is not None
    np.testing.assert_allclose(
        batch.regression_data.targets_system[0], np.arange(15.0)
    )
    np.testing.assert_allclose(
        batch.regression_data.targets_system[1], np.arange(100.0, 115.0)
    )
    assert (batch.regression_data.mask_system == 1).all()
    assert [sd.isomeric_smiles for sd in batch.smiles] == ["C", "O"]
    assert gen.load_stats.n_kept == 2
    # GDB-9 is neutral, closed-shell: charge 0, spin multiplicity (2S+1) = 1.
    assert batch.total_charge == [0.0, 0.0]
    assert batch.multiplicity == [1.0, 1.0]


def test_generator_drops_invalid_smiles(tmp_path: Path) -> None:
    good = Atoms("CH4", positions=np.zeros((5, 3)), info={**_CH4, "SMILES": "C"})
    bad = Atoms("H2O", positions=np.zeros((3, 3)), info={**_H2O, "SMILES": "not_a_smiles"})
    xyz_path = tmp_path / "qm9.xyz"
    ase_write(str(xyz_path), [good, bad], format="extxyz")

    gen = QM9Generator(xyz_file=xyz_path, loading_batch_size=10)
    (batch,) = list(gen)

    assert len(batch) == 1
    assert gen.load_stats.n_raw_rows == 2
    assert gen.load_stats.n_invalid_smiles == 1
    assert gen.load_stats.n_kept == 1


def test_generator_subset_of_tasks(tmp_path: Path) -> None:
    xyz_path = tmp_path / "qm9.xyz"
    _make_qm9_xyz(xyz_path)

    tasks = [QM9Property.mu, QM9Property.gap, QM9Property.Cv]
    gen = QM9Generator(xyz_file=xyz_path, tasks=tasks, loading_batch_size=10)
    (batch,) = list(gen)

    # mu / gap / Cv are at file-columns 3 / 7 / 14 of the arange(15) row.
    np.testing.assert_allclose(
        batch.regression_data.targets_system[0], [3.0, 7.0, 14.0]
    )
    assert gen.task_set().system_map == {"mu": 0, "gap": 1, "Cv": 2}


def test_orchestrator_round_trip_persists_targets_and_smiles(tmp_path: Path) -> None:
    xyz_path = tmp_path / "qm9.xyz"
    _make_qm9_xyz(xyz_path)
    out_dir = tmp_path / "zarr_out"

    gen = QM9Generator(xyz_file=xyz_path, loading_batch_size=10)
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
    assert ds.N_structures == 2
    np.testing.assert_allclose(
        np.asarray(ds.targets_system[0]), np.arange(15.0), rtol=1e-6
    )
    assert (np.asarray(ds.mask_system[:2]) == 1).all()
    assert ds.get_smiles_per_structure()[:2] == ["C", "O"]
    np.testing.assert_allclose(np.asarray(ds.multiplicity[:2]), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(ds.total_charge[:2]), [0.0, 0.0])


def test_nearest_molecule_figures_render() -> None:
    """nearest_molecule_figures renders one grid per embedded query with neighbors."""
    import types

    from matplotlib.figure import Figure

    from threedscriptors.evaluation.retrieval.config import NearestMoleculeTaskConfig
    from threedscriptors.evaluation.retrieval.nearest_molecule import (
        NearestMoleculeResult,
        Neighbor,
        QueryResult,
        nearest_molecule_figures,
    )

    res = NearestMoleculeResult(
        name="nearest_molecule",
        k=2,
        queries=[
            QueryResult(
                query="smiles:CCO",
                embedded=True,
                neighbors=[
                    Neighbor(rank=0, row_index=3, structure_id=3, smiles="CCO", distance=0.0),
                    Neighbor(rank=1, row_index=7, structure_id=7, smiles="CCN", distance=0.12),
                ],
            ),
            # Not embedded -> skipped.
            QueryResult(query="smiles:bad", embedded=False, neighbors=[]),
            # Index query -> SMILES resolved from the store.
            QueryResult(
                query="index:1",
                embedded=True,
                neighbors=[
                    Neighbor(rank=0, row_index=2, structure_id=2, smiles="c1ccccc1", distance=0.05),
                ],
            ),
        ],
    )
    store = types.SimpleNamespace(smiles=["C", "CC", "CCC"])
    cfg = NearestMoleculeTaskConfig(k=2, n_visualize=10)

    figs = nearest_molecule_figures(store, res, cfg)

    assert len(figs) == 2  # the embedded-with-neighbors queries only
    assert all(isinstance(f.figure, Figure) for f in figs)
    names = {str(f.file_name) for f in figs}
    assert names == {
        "retrieval/nearest_molecule/query_000.png",
        "retrieval/nearest_molecule/query_002.png",
    }


def test_nearest_molecule_figures_disabled_by_default() -> None:
    import types

    from threedscriptors.evaluation.retrieval.config import NearestMoleculeTaskConfig
    from threedscriptors.evaluation.retrieval.nearest_molecule import (
        NearestMoleculeResult,
        Neighbor,
        QueryResult,
        nearest_molecule_figures,
    )

    res = NearestMoleculeResult(
        name="nearest_molecule",
        k=1,
        queries=[
            QueryResult(
                query="smiles:CCO",
                embedded=True,
                neighbors=[Neighbor(rank=0, row_index=0, structure_id=0, smiles="CCO", distance=0.0)],
            )
        ],
    )
    store = types.SimpleNamespace(smiles=["CCO"])
    # n_visualize defaults to 0 -> no figures.
    assert nearest_molecule_figures(store, res, NearestMoleculeTaskConfig(k=1)) == []
