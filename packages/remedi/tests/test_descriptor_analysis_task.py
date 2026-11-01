"""``DescriptorAnalysisConfig``: the latent-space analyses run on an explicit eval context.

CPU only: ECFP descriptors over a dataset written by the real ``write_dataset``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from remedi_data.bundle import EvalMetric, LabelColumn
from remedi_data.dataset.tasks import TaskType
from remedi_data.testing import ACHIRAL_TEN_SMILES, write_small_dataset
from remedi_eval.descriptors import EcfpConfig
from remedi_eval.framework import EvalContext, ResourceCache


def test_descriptor_analysis_task_capacity_on_cpu(tmp_path: Path) -> None:
    """``DescriptorAnalysisConfig`` (no longer in the manifest's task union) runs
    on an explicit :class:`EvalContext`, artifacts under ``descriptor_analysis/``."""
    from remedi.latent_evaluation import CapacityDiagnosticTask
    from remedi.latent_evaluation.framework_task import DescriptorAnalysisConfig

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
