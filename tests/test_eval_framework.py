"""Framework runner: fault tolerance, shared resources, decoupled plotting.

CPU-only — ECFP descriptors over benchmark datasets written by the real
``write_dataset``, so the whole framework (manifest -> runner -> tasks ->
artifacts -> status) is exercised without a GPU or a trained model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydantic_yaml as pyd_yaml
import pytest

from remedi.data_handling.bundle import EvalMetric, LabelColumn
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
    render,
    run_manifest,
)
from remedi.evaluation.framework.runner import RunReport
from remedi.evaluation.results import TableResult

from .helpers.bundle_fixtures import ACHIRAL_TEN_SMILES, write_small_dataset

_N_ROWS = len(ACHIRAL_TEN_SMILES)


def _build_reg_zarr(tmp_path: Path, dataset_id: str, targets: np.ndarray) -> Path:
    """One regression dataset at ``tmp_path/datasets/<dataset_id>``."""
    return write_small_dataset(
        tmp_path / "datasets",
        dataset_id=dataset_id,
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.rmse],
        targets=targets,
    )


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


def _failing_panel_task(tmp_path: Path) -> BenchmarkPanelConfig:
    """A valid task whose discovery raises (an unsupported ``format_version``)
    before the panel's own per-dataset fault tolerance can catch anything."""
    broken_root = tmp_path / "broken_root"
    broken = broken_root / "broken_dataset"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "dataset.yaml").write_text("format_version: 99\n")
    (broken / "dataset_config.yaml").write_text("{}\n")
    return BenchmarkPanelConfig(
        eval_root=broken_root, learners=[LinearLearnerConfig(ridge_alpha=1.0)]
    )


def test_keep_going_records_failed_task_but_finishes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.tasks.append(_failing_panel_task(tmp_path))
    report = run_manifest(manifest)  # must not raise
    out = manifest.output_root

    assert report.n_failed == 1
    # The healthy task's artifacts still landed, next to the manifest.
    assert (out / "manifest.yaml").exists()
    csv = pd.read_csv(out / "benchmark" / "results.csv")
    assert set(csv.dataset_id) == {"toy_a", "toy_b"}

    # status.yaml round-trips and records both tasks, the healthy one with its
    # artifact and the broken one with its error.
    status = pyd_yaml.parse_yaml_file_as(RunReport, out / "status.yaml")
    healthy, broken = status.statuses
    assert healthy.ok and healthy.kind == "benchmark_panel"
    assert any("results.csv" in a["file_name"] for a in healthy.artifacts)
    assert not broken.ok
    assert broken.error and broken.traceback


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


def test_descriptor_analysis_task_capacity_on_cpu(tmp_path: Path) -> None:
    """``DescriptorAnalysisConfig`` (no longer in the manifest's task union) runs
    on an explicit :class:`EvalContext`, artifacts under ``descriptor_analysis/``."""
    from remedi.latent_evaluation import CapacityDiagnosticTask
    from remedi.latent_evaluation.framework_task import DescriptorAnalysisConfig

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
        dataset_path=tmp_path / "datasets" / "corpus",
        dataset_id="corpus",
        tasks=[CapacityDiagnosticTask()],
    )
    for artifact in task.run(ctx):
        artifact.serialize_to(out)

    assert (out / "descriptor_analysis" / "capacity_diagnostic.yaml").exists()


def test_benchmark_plotter_renders_from_a_loaded_results_table(tmp_path: Path) -> None:
    """Plotting is decoupled: the registered plotter runs on a reloaded artifact."""
    import matplotlib

    matplotlib.use("Agg")
    import remedi.evaluation.framework.builtin_plotters  # noqa: F401

    frame = pd.DataFrame(
        {
            "dataset_id": ["esol", "esol", "bace", "bace"],
            "learner_kind": ["linear", "mlp", "linear", "mlp"],
            "metric_name": ["RMSE", "RMSE", "AUROC", "AUROC"],
            "metric_value": [1.1, 0.9, 0.82, 0.85],
        }
    )
    TableResult(file_name=Path("results.csv"), frame=frame).serialize_to(tmp_path)
    loaded = TableResult.load(tmp_path / "results.csv")
    pd.testing.assert_frame_equal(loaded, frame)

    figures = render("benchmark_results", loaded, tmp_path / "plots")
    # one figure per metric_name (RMSE, AUROC)
    assert len(figures) == 2
    assert (tmp_path / "plots" / "benchmark_RMSE.png").exists()
    assert (tmp_path / "plots" / "benchmark_AUROC.png").exists()
