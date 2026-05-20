"""End-to-end runner test on a hand-built synthetic zarr.

Hand-builds two tiny benchmark zarrs (regression + binary classification),
each with a real ``MoleculeDataset`` + ``BenchmarkManifest`` sidecar, then runs
the eval runner against both. Pins: discovery walks the root, descriptor x
learner cross-product produces the right number of rows, dispatch lands on
the matching task-type method, and ``results.csv`` + ``results.yaml`` are
written. The pipeline-stage ingest is skipped — we hand-write rows via
ShardAlignedWriter to keep the test fast (no RDKit ETKDG / MMFF).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from threedscriptors.configuration.dataset_config import DatasetConfig
from threedscriptors.data_handling.benchmarks import (
    BenchmarkManifest,
    EvalMetric,
    SplitVariant,
)
from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from threedscriptors.data_handling.dataset_creation.shard_aligned_writer import (
    ShardAlignedWriter,
)
from threedscriptors.evaluation.benchmark.descriptors import EcfpConfig
from threedscriptors.evaluation.benchmark.learners import LinearLearnerConfig
from threedscriptors.evaluation.benchmark.runner import (
    BenchmarkResultRow,
    EvalConfig,
    run_eval,
)


def _hand_build_zarr(
    path: Path,
    smiles: list[str],
    targets: np.ndarray,
    split_codes: np.ndarray,
    task_type: TaskType,
    metric: EvalMetric,
    source: str,
) -> None:
    """Write a minimal one-task zarr + manifest with the orchestrator bypassed.

    Two heavy stages (conformer generation, embedding) are skipped: we write
    zero positions + carbon atoms purely to satisfy the writer's invariants.
    The eval path only reads positions for structure-based descriptors, so the
    ECFP runner exercised here never touches them.
    """
    n = len(smiles)
    atoms_per = 3
    cfg = DatasetConfig(
        atom_chunk=8,
        molecule_chunk=4,
        atom_chunks_per_shard=4,
        molecule_chunks_per_shard=4,
        contains_smiles=True,
        tasks=TaskSet.from_list(
            [TaskConfig(name="y", task_type=task_type, scope=TaskScope.system)]
        ),
    )
    ds = MoleculeDataset.create_empty_dataset(path, cfg)
    smiles_to_id = ds.smiles.append_new_lines(smiles)
    iso_to_id = ds.isomeric_smiles.append_new_lines(smiles)
    mol_ids = np.array([smiles_to_id[s] for s in smiles], dtype="i8")
    iso_ids = np.array([iso_to_id[s] for s in smiles], dtype="i8")
    writer = ShardAlignedWriter(ds)
    writer.append_batch(
        positions=np.zeros((n * atoms_per, 3), dtype="f4"),
        atomic_numbers=np.full(n * atoms_per, 6, dtype="u1"),
        batch_ptr_cumsum=np.arange(1, n + 1) * atoms_per,
        molecule_ids=mol_ids,
        stereoisomer_ids=iso_ids,
        total_charge=np.zeros(n, dtype="f4"),
        multiplicity=np.ones(n, dtype="f4"),
        system_targets=np.asarray(targets, dtype="f4").reshape(-1, 1),
        system_masks=np.ones((n, 1), dtype="u1"),
        atom_targets=None,
        atom_masks=None,
        split=np.asarray(split_codes, dtype="u1"),
    )
    ds.smiles.close()
    ds.isomeric_smiles.close()
    writer.finalize()
    BenchmarkManifest(
        dataset_id=path.name,
        metric=metric,
        split_variant=SplitVariant.scaffold,
        source=source,  # type: ignore[arg-type]
    ).to_zarr_dir(path)


_SMILES_10 = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "OC", "CC", "CCC", "CCCC",
    "CCCCC", "CCCCCC",
]
# 6 train (0), 2 valid (1), 2 test (2)
_SPLIT_10 = np.array([0] * 6 + [1] * 2 + [2] * 2, dtype="u1")


def test_run_eval_regression_writes_single_row(tmp_path: Path) -> None:
    eval_root = tmp_path / "datasets"
    eval_root.mkdir()
    targets = np.array([0.1, 0.5, 0.3, 0.4, 0.2, 0.15, 0.18, 0.22, 0.27, 0.3])
    _hand_build_zarr(
        eval_root / "toy_reg",
        _SMILES_10,
        targets,
        _SPLIT_10,
        task_type=TaskType.regression,
        metric=EvalMetric.rmse,
        source="moleculenet",
    )

    out_dir = tmp_path / "eval_out"
    cfg = EvalConfig(
        eval_root=eval_root,
        output_dir=out_dir,
        descriptors=[EcfpConfig(name="ecfp_512", length=512)],
        learners=[LinearLearnerConfig(ridge_alpha=1.0)],
    )
    rows = run_eval(cfg)

    assert len(rows) == 1
    (row,) = rows
    assert isinstance(row, BenchmarkResultRow)
    assert row.dataset_id == "toy_reg"
    assert row.descriptor_name == "ecfp_512"
    assert row.learner_kind == "linear"
    assert row.metric_name == "RMSE"
    assert np.isfinite(row.metric_value)
    assert (row.n_train, row.n_val, row.n_test) == (6, 2, 2)

    csv = pd.read_csv(out_dir / "results.csv")
    assert len(csv) == 1
    assert (out_dir / "results.yaml").exists()


def test_run_eval_binary_classification_dispatch(tmp_path: Path) -> None:
    eval_root = tmp_path / "datasets"
    eval_root.mkdir()
    # Alternating labels so the (8, 9) test fold has both classes present —
    # otherwise sklearn's AUROC raises (only-one-class-in-y_true).
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    _hand_build_zarr(
        eval_root / "toy_cls",
        _SMILES_10,
        labels,
        _SPLIT_10,
        task_type=TaskType.classification,
        metric=EvalMetric.auroc,
        source="tdc",
    )

    out_dir = tmp_path / "eval_out"
    cfg = EvalConfig(
        eval_root=eval_root,
        output_dir=out_dir,
        descriptors=[EcfpConfig(name="ecfp_256", length=256)],
        learners=[LinearLearnerConfig(logreg_C=1.0)],
    )
    rows = run_eval(cfg)

    assert len(rows) == 1
    (row,) = rows
    assert row.source == "tdc"
    assert row.metric_name == "AUROC"
    # The test fold has 2 rows (one class 0, one class 1) — AUROC is finite.
    assert np.isfinite(row.metric_value)


def test_run_eval_cross_product_two_descriptors_two_learners(tmp_path: Path) -> None:
    eval_root = tmp_path / "datasets"
    eval_root.mkdir()
    targets = np.linspace(0.0, 1.0, 10)
    _hand_build_zarr(
        eval_root / "toy_reg",
        _SMILES_10,
        targets,
        _SPLIT_10,
        task_type=TaskType.regression,
        metric=EvalMetric.rmse,
        source="moleculenet",
    )

    cfg = EvalConfig(
        eval_root=eval_root,
        output_dir=tmp_path / "eval_out",
        descriptors=[
            EcfpConfig(name="ecfp_128", length=128),
            EcfpConfig(name="ecfp_512", length=512),
        ],
        learners=[
            LinearLearnerConfig(ridge_alpha=0.1),
            LinearLearnerConfig(ridge_alpha=10.0),
        ],
    )
    rows = run_eval(cfg)
    # 1 benchmark x 2 descriptors x 2 learners = 4 rows.
    assert len(rows) == 4
    names = {(r.descriptor_name, r.learner_kind) for r in rows}
    assert names == {
        ("ecfp_128", "linear"),
        ("ecfp_512", "linear"),
    }
