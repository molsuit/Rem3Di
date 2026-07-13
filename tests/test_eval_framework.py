"""Framework runner: fault tolerance, shared resources, decoupled plotting.

CPU-only — uses ECFP descriptors over hand-built synthetic benchmark zarrs, so
the whole framework (manifest -> runner -> tasks -> artifacts -> status) is
exercised without a GPU / a trained model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydantic_yaml as pyd_yaml
import pytest

from remedi.configuration.dataset_config import DatasetConfig
from remedi.data_handling.benchmarks import (
    BenchmarkManifest,
    EvalMetric,
    SplitVariant,
)
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from remedi.data_handling.dataset_creation.shard_aligned_writer import (
    ShardAlignedWriter,
)
from remedi.evaluation.benchmark.descriptors import EcfpConfig
from remedi.evaluation.benchmark.learners import LinearLearnerConfig
from remedi.evaluation.framework import (
    BenchmarkPanelConfig,
    EmbeddingSpec,
    EvalManifest,
    ResourceCache,
    register_plotter,
    render,
    run_manifest,
)
from remedi.evaluation.framework.runner import RunReport
from remedi.evaluation.results import ArrayResult, FigureResult, TableResult

_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "OC",
    "CC",
    "CCC",
    "CCCC",
    "CCCCC",
    "CCCCCC",
]
_SPLIT = np.array([0] * 6 + [1] * 2 + [2] * 2, dtype="u1")


def _build_reg_zarr(path: Path, targets: np.ndarray) -> None:
    n, atoms_per = len(_SMILES), 3
    cfg = DatasetConfig(
        atom_chunk=8,
        molecule_chunk=4,
        atom_chunks_per_shard=4,
        molecule_chunks_per_shard=4,
        contains_smiles=True,
        tasks=TaskSet.from_list(
            [
                TaskConfig(
                    name="y", task_type=TaskType.regression, scope=TaskScope.system
                )
            ]
        ),
    )
    ds = MoleculeDataset.create_empty_dataset(path, cfg)
    s2i = ds.smiles.append_new_lines(_SMILES)
    i2i = ds.isomeric_smiles.append_new_lines(_SMILES)
    writer = ShardAlignedWriter(ds)
    writer.append_batch(
        positions=np.zeros((n * atoms_per, 3), dtype="f4"),
        atomic_numbers=np.full(n * atoms_per, 6, dtype="u1"),
        batch_ptr_cumsum=np.arange(1, n + 1) * atoms_per,
        molecule_ids=np.array([s2i[s] for s in _SMILES], dtype="i8"),
        stereoisomer_ids=np.array([i2i[s] for s in _SMILES], dtype="i8"),
        total_charge=np.zeros(n, dtype="f4"),
        multiplicity=np.ones(n, dtype="f4"),
        system_targets=np.asarray(targets, dtype="f4").reshape(-1, 1),
        system_masks=np.ones((n, 1), dtype="u1"),
        atom_targets=None,
        atom_masks=None,
        split=_SPLIT,
    )
    ds.smiles.close()
    ds.isomeric_smiles.close()
    writer.finalize()
    BenchmarkManifest(
        dataset_id=path.name,
        metric=EvalMetric.rmse,
        split_variant=SplitVariant.scaffold,
        source="moleculenet",  # type: ignore[arg-type]
    ).to_zarr_dir(path)


def _manifest(tmp_path: Path) -> EvalManifest:
    eval_root = tmp_path / "datasets"
    eval_root.mkdir()
    _build_reg_zarr(eval_root / "toy_a", np.linspace(0, 1, 10))
    _build_reg_zarr(eval_root / "toy_b", np.linspace(1, 0, 10))
    return EvalManifest(
        model=EcfpConfig(name="ecfp_256", length=256),
        output_root=tmp_path / "eval_out" / "model_x",
        tasks=[
            BenchmarkPanelConfig(
                eval_root=eval_root, learners=[LinearLearnerConfig(ridge_alpha=1.0)]
            )
        ],
    )


def test_run_manifest_writes_results_status_and_manifest(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    report = run_manifest(manifest)
    out = manifest.output_root

    assert report.n_failed == 0
    assert (out / "manifest.yaml").exists()
    assert (out / "benchmark" / "results.csv").exists()
    csv = pd.read_csv(out / "benchmark" / "results.csv")
    assert set(csv.dataset_id) == {"toy_a", "toy_b"}

    # status.yaml round-trips and records the task as ok with its artifact.
    status = pyd_yaml.parse_yaml_file_as(RunReport, out / "status.yaml")
    assert len(status.statuses) == 1
    (st,) = status.statuses
    assert st.ok and st.kind == "benchmark_panel"
    assert any("results.csv" in a["file_name"] for a in st.artifacts)


def _failing_retrieval_task(tmp_path: Path):
    """A valid union task that raises at run time (dataset path doesn't exist)."""
    from remedi.evaluation.framework import RetrievalConfig
    from remedi.evaluation.retrieval.config import (
        TanimotoSimilarityTaskConfig,
    )

    return RetrievalConfig(
        dataset_path=tmp_path / "does_not_exist",
        dataset_id="missing",
        tasks=[TanimotoSimilarityTaskConfig(k=3)],
    )


def test_keep_going_records_failed_task_but_finishes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.tasks.append(_failing_retrieval_task(tmp_path))
    report = run_manifest(manifest)  # must not raise

    assert report.n_failed == 1
    by_kind = {s.kind: s for s in report.statuses}
    assert by_kind["benchmark_panel"].ok
    assert not by_kind["retrieval"].ok
    assert by_kind["retrieval"].error and by_kind["retrieval"].traceback
    # The healthy task's artifacts still landed.
    assert (manifest.output_root / "benchmark" / "results.csv").exists()


def test_fail_fast_raises(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.keep_going = False
    manifest.tasks.append(_failing_retrieval_task(tmp_path))
    with pytest.raises(Exception):  # noqa: B017 - any error from the failing task
        run_manifest(manifest)
    # status.yaml was still flushed in the finally block.
    assert (manifest.output_root / "status.yaml").exists()


def test_shared_resource_built_once(tmp_path: Path) -> None:
    eval_root = tmp_path / "datasets"
    eval_root.mkdir()
    _build_reg_zarr(eval_root / "toy_a", np.linspace(0, 1, 10))
    ds = MoleculeDataset.open_existing_dataset_from_dir(eval_root / "toy_a")
    spec = EmbeddingSpec(
        dataset_id="toy_a",
        descriptor=EcfpConfig(name="ecfp_256", length=256),
        dataset=ds,
        cache_dir=tmp_path / "cache",
    )
    cache = ResourceCache()
    X1 = cache.get(spec)
    X2 = cache.get(spec)  # same key -> served from memo, not rebuilt
    assert X1 is X2
    assert cache.build_count(spec) == 1


def test_table_and_array_artifacts_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    TableResult(file_name=Path("t.csv"), frame=df).serialize_to(tmp_path)
    pd.testing.assert_frame_equal(TableResult.load(tmp_path / "t.csv"), df)

    arrs = {"coords": np.arange(6.0).reshape(3, 2), "color": np.array([0, 1, 2])}
    ArrayResult(file_name=Path("a.npz"), arrays=arrs).serialize_to(tmp_path)
    loaded = ArrayResult.load(tmp_path / "a.npz")
    np.testing.assert_array_equal(loaded["coords"], arrs["coords"])


def test_plotter_registry_renders_from_loaded_artifact(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    @register_plotter("demo_table")
    def _plot(df: pd.DataFrame, out_dir: Path) -> list[FigureResult]:
        fig, ax = plt.subplots()
        ax.plot(df["a"])
        return [FigureResult(file_name=Path("demo.png"), figure=fig)]

    df = pd.DataFrame({"a": [3, 1, 2]})
    TableResult(file_name=Path("demo.csv"), frame=df).serialize_to(tmp_path)
    payload = TableResult.load(tmp_path / "demo.csv")
    figs = render("demo_table", payload, tmp_path / "plots")
    assert len(figs) == 1
    assert (tmp_path / "plots" / "demo.png").exists()


def test_retrieval_task_tanimoto_on_cpu(tmp_path: Path) -> None:
    """Retrieval task end-to-end on CPU: an ECFP 'model' as the embedding, the
    Tanimoto sub-task sharing one EmbeddingSpec + IndexSpec. (Nearest-molecule
    needs a GPU re-embedder, so it is exercised in the GPU smoke, not here.)"""
    from remedi.evaluation.framework import RetrievalConfig
    from remedi.evaluation.retrieval.config import (
        TanimotoSimilarityTaskConfig,
    )

    eval_root = tmp_path / "datasets"
    eval_root.mkdir()
    _build_reg_zarr(eval_root / "corpus", np.linspace(0, 1, 10))

    manifest = EvalManifest(
        model=EcfpConfig(name="ecfp_256", length=256),
        output_root=tmp_path / "eval_out" / "model_x",
        tasks=[
            RetrievalConfig(
                dataset_path=eval_root / "corpus",
                dataset_id="corpus",
                tasks=[
                    TanimotoSimilarityTaskConfig(
                        k=3, n_query_sample=5, n_global_pairs=20
                    )
                ],
            )
        ],
    )
    report = run_manifest(manifest)
    out = manifest.output_root
    assert report.n_failed == 0
    assert (out / "retrieval" / "tanimoto_similarity.yaml").exists()
    assert (out / "retrieval" / "retrieval_report.yaml").exists()
    assert (out / "retrieval" / "tanimoto_results.csv").exists()


def test_descriptor_analysis_task_capacity_on_cpu(tmp_path: Path) -> None:
    """Descriptor-analysis task end-to-end on CPU: capacity diagnostic over an
    ECFP embedding, artifacts re-rooted under descriptor_analysis/."""
    from remedi.evaluation.descriptor_analysis import CapacityDiagnosticTask
    from remedi.evaluation.framework import DescriptorAnalysisConfig

    eval_root = tmp_path / "datasets"
    eval_root.mkdir()
    _build_reg_zarr(eval_root / "corpus", np.linspace(0, 1, 10))

    manifest = EvalManifest(
        model=EcfpConfig(name="ecfp_256", length=256),
        output_root=tmp_path / "eval_out" / "model_x",
        tasks=[
            DescriptorAnalysisConfig(
                dataset_path=eval_root / "corpus",
                dataset_id="corpus",
                tasks=[CapacityDiagnosticTask()],
            )
        ],
    )
    report = run_manifest(manifest)
    out = manifest.output_root
    assert report.n_failed == 0
    assert (out / "descriptor_analysis" / "capacity_diagnostic.yaml").exists()


def test_benchmark_plotter_renders_from_results_csv(tmp_path: Path) -> None:
    import remedi.evaluation.framework.builtin_plotters  # noqa: F401
    from remedi.evaluation.framework.plotting import render

    df = pd.DataFrame(
        {
            "dataset_id": ["esol", "esol", "bace", "bace"],
            "learner_kind": ["linear", "mlp", "linear", "mlp"],
            "metric_name": ["RMSE", "RMSE", "AUROC", "AUROC"],
            "metric_value": [1.1, 0.9, 0.82, 0.85],
        }
    )
    figs = render("benchmark_results", df, tmp_path / "plots")
    # one figure per metric_name (RMSE, AUROC)
    assert len(figs) == 2
    assert (tmp_path / "plots" / "benchmark_RMSE.png").exists()
    assert (tmp_path / "plots" / "benchmark_AUROC.png").exists()
