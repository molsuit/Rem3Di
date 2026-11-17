"""``DescriptorAnalysisConfig``: the latent-space analyses run on an explicit eval context.

CPU only: ECFP descriptors over a dataset written by the real ``write_dataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pydantic
from remedi_data.bundle import EvalMetric, LabelColumn, TaskType

from remedi_eval.descriptors import EcfpConfig
from remedi_eval.framework import EvalContext
from remedi_eval.projection import PcaConfig, ProjectionConfig, UmapConfig
from testing import ACHIRAL_TEN_SMILES, write_small_dataset


def test_descriptor_analysis_task_capacity_on_cpu(tmp_path: Path) -> None:
    """``DescriptorAnalysisConfig`` runs on an explicit :class:`EvalContext`,
    artifacts under ``descriptor_analysis/``; the dataset id comes from its spec."""
    from remedi_eval.framework.tasks.descriptor_analysis import CapacityDiagnosticTask
    from remedi_eval.framework.tasks.descriptor_analysis.task import (
        DescriptorAnalysisConfig,
    )

    write_small_dataset(
        tmp_path / "datasets",
        dataset_id="corpus",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.rmse],
        targets=np.linspace(0, 1, len(ACHIRAL_TEN_SMILES)),
    )

    out = tmp_path / "eval_out" / "model_x"
    out.mkdir(parents=True)
    ctx = EvalContext(
        output_root=out,
        descriptor_cache_dir=tmp_path / "cache",
        model=EcfpConfig(name="ecfp_256", fingerprint_size=256),
    )
    task = DescriptorAnalysisConfig(
        dataset_path=tmp_path / "datasets" / "corpus",
        tasks=[CapacityDiagnosticTask()],
    )
    for artifact in task.run(ctx):
        artifact.serialize_to(out)

    assert (out / "descriptor_analysis" / "capacity_diagnostic.yaml").exists()


def test_unknown_analysis_fields_are_rejected() -> None:
    """A misspelt option fails instead of silently running the default analysis."""
    import pytest
    from pydantic import ValidationError

    from remedi_eval.framework.tasks.descriptor_analysis.task import (
        DescriptorAnalysisConfig,
    )

    with pytest.raises(ValidationError, match="dead_thr"):
        DescriptorAnalysisConfig.model_validate(
            {
                "dataset_path": "corpus",
                "tasks": [{"kind": "capacity_diagnostic", "dead_thr": 0.1}],
            }
        )


def test_an_eval_manifest_runs_descriptor_analyses_of_its_model(
    tmp_path: Path,
) -> None:
    """One manifest type serves every run: ``run_manifest`` analyses the
    manifest's model (here ECFP) and records the task in ``status.yaml``."""
    from remedi_eval.framework import EvalManifest, run_manifest

    write_small_dataset(
        tmp_path / "datasets",
        dataset_id="corpus",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.rmse],
        targets=np.linspace(0, 1, len(ACHIRAL_TEN_SMILES)),
    )
    manifest = EvalManifest.model_validate(
        {
            "model": {"kind": "ecfp", "name": "ecfp_256", "fingerprint_size": 256},
            "output_root": str(tmp_path / "eval_out" / "ecfp_256"),
            "tasks": [
                {
                    "kind": "descriptor_analysis",
                    "dataset_path": str(tmp_path / "datasets" / "corpus"),
                    "tasks": [{"kind": "descriptor_distribution"}],
                }
            ],
        }
    )
    report = run_manifest(manifest)

    assert report.n_failed == 0
    analysis_dir = manifest.output_root / "descriptor_analysis"
    assert (analysis_dir / "descriptor_distribution_summary.yaml").exists()


def test_an_analysis_without_its_extra_names_the_extra(
    tmp_path: Path, monkeypatch
) -> None:
    """Without chemiscope installed the chemiscope export fails naming the
    ``descriptor-analysis`` extra; the other analyses do not need it."""
    import sys

    import pytest

    from remedi_eval.framework.tasks.descriptor_analysis import (
        ChemiscopeProjectionTask,
        DescriptorAnalysisContext,
    )
    from remedi_eval.framework.tasks.descriptor_analysis.tmqm_chemical_features import (
        compute_metal_environment_features,
    )

    monkeypatch.setitem(sys.modules, "chemiscope", None)
    monkeypatch.setitem(sys.modules, "pymatgen", None)
    monkeypatch.setitem(sys.modules, "pymatgen.core", None)
    ctx = DescriptorAnalysisContext.build(
        dataset=cast("Any", None),
        descriptors=np.eye(4),
        projection_config=PcaConfig(),
    )
    with pytest.raises(ImportError, match=r"remedi-eval\[descriptor-analysis\]"):
        ChemiscopeProjectionTask().run(ctx)

    from ase import Atoms

    with pytest.raises(ImportError, match=r"remedi-eval\[descriptor-analysis\]"):
        compute_metal_environment_features(
            [Atoms("FeH", positions=[[0, 0, 0], [1.5, 0, 0]])]
        )


def _write_corpus(root: Path) -> Path:
    write_small_dataset(
        root,
        dataset_id="corpus",
        labels=[LabelColumn(name="y", task_type=TaskType.regression)],
        metrics=[EvalMetric.rmse],
        targets=np.linspace(0, 1, len(ACHIRAL_TEN_SMILES)),
    )
    return root / "corpus"


def _context(tmp_path: Path) -> tuple[EvalContext, Path]:
    out = tmp_path / "eval_out" / "model_x"
    out.mkdir(parents=True)
    return (
        EvalContext(
            output_root=out,
            descriptor_cache_dir=tmp_path / "cache",
            model=EcfpConfig(name="ecfp_256", fingerprint_size=256),
        ),
        out,
    )


def test_max_structures_cuts_every_per_structure_analysis(
    tmp_path: Path, monkeypatch
) -> None:
    """With ``max_structures`` every analysis sees the same first rows: colours,
    chemiscope frames, top-norm panels and the scorecard's chemistry all have
    as many entries as there are descriptor rows, read from the dataset once."""
    from remedi_data.dataset.molecule_dataset import MoleculeDataset
    from remedi_eval.framework.tasks.descriptor_analysis import (
        ChemiscopeClusterTask,
        ChemiscopeProjectionTask,
        DescriptorStructureBenchmarkTask,
        NumAtomsColor,
        ProjectionPlotTask,
        RegressionTargetColor,
        TopNormDescriptorsTask,
    )
    from remedi_eval.framework.tasks.descriptor_analysis.task import (
        DescriptorAnalysisConfig,
    )

    dataset_path = _write_corpus(tmp_path / "datasets")
    ctx, out = _context(tmp_path)

    reads: list[int | None] = []
    original = MoleculeDataset.get_all_molecules

    def counting_get_all_molecules(self, *args, **kwargs):
        reads.append(args[0] if args else kwargs.get("N_molecules"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        MoleculeDataset, "get_all_molecules", counting_get_all_molecules
    )

    task = DescriptorAnalysisConfig(
        dataset_path=dataset_path,
        max_structures=6,
        projection=PcaConfig(),
        tasks=[
            ProjectionPlotTask(file_name="atoms.png", color_provider=NumAtomsColor()),
            ProjectionPlotTask(
                file_name="target.png", color_provider=RegressionTargetColor()
            ),
            ChemiscopeProjectionTask(),
            TopNormDescriptorsTask(n_top=3),
            ChemiscopeClusterTask(
                cluster_projection=None, min_cluster_sizes=[2], default_color_mcs=2
            ),
            DescriptorStructureBenchmarkTask(
                cluster_projection=None,
                canonical_min_cluster_size=2,
                sweep_min_cluster_sizes=[2],
                pure_island_min_size=2,
            ),
        ],
    )
    names = set()
    for artifact in task.run(ctx):
        artifact.serialize_to(out)
        names.add(artifact.file_name.name)

    assert {
        "atoms.png",
        "target.png",
        "chemiscope_projection.json.gz",
        "top_norm_descriptors.png",
        "chemiscope_clusters.json.gz",
        "descriptor_structure_benchmark.yaml",
    } <= names
    assert reads == [6]


def test_a_failed_analysis_fails_the_task(tmp_path: Path) -> None:
    """The other analyses still write their artifacts, then the task raises with
    every failure, so ``status.yaml`` records it and ``keep_going=False`` stops."""
    import pytest

    from remedi_eval.framework.tasks.descriptor_analysis import (
        CapacityDiagnosticTask,
        NumAtomsColor,
        ProjectionPlotTask,
    )
    from remedi_eval.framework.tasks.descriptor_analysis.task import (
        DescriptorAnalysisConfig,
        DescriptorAnalysisError,
    )

    dataset_path = _write_corpus(tmp_path / "datasets")
    ctx, out = _context(tmp_path)
    task = DescriptorAnalysisConfig(
        dataset_path=dataset_path,
        tasks=[
            # No projection configured: the plot cannot run.
            ProjectionPlotTask(file_name="atoms.png", color_provider=NumAtomsColor()),
            CapacityDiagnosticTask(),
        ],
    )
    written = []
    with pytest.raises(DescriptorAnalysisError, match="ProjectionPlotTask"):
        for artifact in task.run(ctx):
            artifact.serialize_to(out)
            written.append(artifact.file_name.name)
    assert "capacity_diagnostic.yaml" in written


def test_pca_projection_has_the_requested_components_and_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    descriptors = rng.normal(size=(50, 20)) + 3.0
    config = PcaConfig(n_components=10)
    first = config.project(descriptors)
    assert first.shape == (50, 10)
    np.testing.assert_allclose(first, config.project(descriptors))
    np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1e-10)


def test_the_projection_union_reads_its_kind() -> None:
    adapter = pydantic.TypeAdapter(ProjectionConfig)
    assert adapter.validate_python({"kind": "pca"}) == PcaConfig()
    umap_config = adapter.validate_python({"kind": "umap", "n_components": 15})
    assert umap_config == UmapConfig(n_components=15)
    assert umap_config.metric == "cosine" and umap_config.min_dist == 0.0


def test_capacity_plot_draws_the_configured_dead_threshold() -> None:
    from remedi_eval.framework.tasks.descriptor_analysis import CapacityDiagnosticTask
    from remedi_eval.framework.tasks.descriptor_analysis.context import (
        DescriptorAnalysisContext,
    )
    from remedi_eval.results import FigureResult

    rng = np.random.default_rng(0)
    descriptors = rng.normal(size=(200, 6))
    ctx = DescriptorAnalysisContext.build(
        dataset=cast("Any", None), descriptors=descriptors
    )
    results = CapacityDiagnosticTask(dead_threshold=0.35).run(ctx)
    (figure,) = [
        r.figure
        for r in results
        if isinstance(r, FigureResult) and r.file_name.name.endswith("per_dim.png")
    ]
    threshold_lines = [
        line
        for axis in figure.axes
        for line in axis.get_lines()
        if line.get_label() == "dead threshold"
    ]
    assert threshold_lines
    assert all(np.allclose(line.get_ydata(), 0.35) for line in threshold_lines)


def test_capacity_entropy_has_no_invalid_value_warnings() -> None:
    """Empty histogram bins must not produce ``0 * log(0)`` warnings."""
    import warnings

    from remedi_eval.framework.tasks.descriptor_analysis.capacity_diagnostic import (
        run_latent_space_capacity_diagnostic,
    )

    rng = np.random.default_rng(0)
    descriptors = np.column_stack(
        [rng.normal(size=500), np.zeros(500), rng.exponential(size=500)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = run_latent_space_capacity_diagnostic(descriptors, bins=64)
    assert report.H_per_dim[1] == 0.0
    assert report.dead_dims >= 1
