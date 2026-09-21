"""Framework runner: fault tolerance, shared resources, decoupled plotting.

CPU-only — ECFP descriptors over benchmark zarrs built by the real
``ingest_benchmark`` prepare task from real bundles, so the whole framework
(manifest -> runner -> tasks -> artifacts -> status) is exercised without a GPU
or a trained model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydantic_yaml as pyd_yaml
import pytest

from remedi.data_handling.bundle import BenchmarkTask
from remedi.data_handling.dataset.molecule_dataset import MoleculeDataset
from remedi.data_handling.dataset.tasks import TaskType
from remedi.evaluation.benchmark.descriptors import EcfpConfig
from remedi.evaluation.benchmark.learners import LinearLearnerConfig
from remedi.evaluation.framework import (
    BenchmarkPanelConfig,
    EmbeddingSpec,
    EvalContext,
    EvalManifest,
    ResourceCache,
    register_plotter,
    render,
    run_manifest,
)
from remedi.evaluation.framework.runner import RunReport
from remedi.evaluation.results import ArrayResult, FigureResult, TableResult

from .helpers.bundle_fixtures import (
    ACHIRAL_TEN_SMILES,
    ingest_tiny_bundle,
    write_conformers_bundle,
)

_N_ROWS = len(ACHIRAL_TEN_SMILES)


def _build_reg_zarr(tmp_path: Path, dataset_id: str, targets: np.ndarray) -> Path:
    """One regression bundle ingested into ``tmp_path/datasets/<dataset_id>``."""
    write_conformers_bundle(
        tmp_path / "bundles",
        dataset_id=dataset_id,
        tasks=[BenchmarkTask(name="y", task_type=TaskType.regression)],
        metrics=["RMSE"],
        targets=targets,
    )
    return ingest_tiny_bundle(tmp_path / "bundles", tmp_path / "datasets", dataset_id)


def _manifest(tmp_path: Path) -> EvalManifest:
    eval_root = tmp_path / "datasets"
    _build_reg_zarr(tmp_path, "toy_a", np.linspace(0, 1, _N_ROWS))
    _build_reg_zarr(tmp_path, "toy_b", np.linspace(1, 0, _N_ROWS))
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


def _failing_panel_task(tmp_path: Path) -> BenchmarkPanelConfig:
    """A valid union task that raises at run time.

    Its eval root holds a directory that *looks* like an ingested benchmark
    zarr but whose ``benchmark.yaml`` does not parse, so discovery raises before
    the panel's own per-dataset fault tolerance can catch anything.
    """
    broken_root = tmp_path / "broken_root"
    broken = broken_root / "broken_dataset"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "benchmark.yaml").write_text("format_version: 99\n")
    (broken / "dataset_config.yaml").write_text("{}\n")
    return BenchmarkPanelConfig(
        eval_root=broken_root, learners=[LinearLearnerConfig(ridge_alpha=1.0)]
    )


def test_keep_going_records_failed_task_but_finishes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.tasks.append(_failing_panel_task(tmp_path))
    report = run_manifest(manifest)  # must not raise

    assert report.n_failed == 1
    healthy, broken = report.statuses
    assert healthy.ok and healthy.kind == "benchmark_panel"
    assert not broken.ok
    assert broken.error and broken.traceback
    # The healthy task's artifacts still landed.
    assert (manifest.output_root / "benchmark" / "results.csv").exists()


def test_fail_fast_raises(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.keep_going = False
    manifest.tasks.append(_failing_panel_task(tmp_path))
    with pytest.raises(Exception):  # noqa: B017 - any error from the failing task
        run_manifest(manifest)
    # status.yaml was still flushed in the finally block.
    assert (manifest.output_root / "status.yaml").exists()


def test_shared_resource_built_once(tmp_path: Path) -> None:
    zarr_path = _build_reg_zarr(tmp_path, "toy_a", np.linspace(0, 1, _N_ROWS))
    ds = MoleculeDataset.open_existing_dataset_from_dir(zarr_path)
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


def test_descriptor_analysis_task_capacity_on_cpu(tmp_path: Path) -> None:
    """The moved descriptor-analysis task still runs on a framework context.

    ``DescriptorAnalysisConfig`` left the ``TaskConfig`` union in step 8 (it now
    lives in :mod:`remedi.latent_evaluation`), so it is driven here through an
    explicit :class:`EvalContext` instead of a manifest. Capacity diagnostic
    over an ECFP embedding, artifacts re-rooted under ``descriptor_analysis/``.
    """
    from remedi.latent_evaluation import CapacityDiagnosticTask
    from remedi.latent_evaluation.framework_task import DescriptorAnalysisConfig

    eval_root = tmp_path / "datasets"
    _build_reg_zarr(tmp_path, "corpus", np.linspace(0, 1, _N_ROWS))

    out = tmp_path / "eval_out" / "model_x"
    out.mkdir(parents=True)
    ctx = EvalContext(
        output_root=out,
        resource_cache_dir=tmp_path / "cache",
        resources=ResourceCache(),
        model=EcfpConfig(name="ecfp_256", length=256),
    )
    task = DescriptorAnalysisConfig(
        dataset_path=eval_root / "corpus",
        dataset_id="corpus",
        tasks=[CapacityDiagnosticTask()],
    )
    for artifact in task.run(ctx):
        artifact.serialize_to(out)

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
