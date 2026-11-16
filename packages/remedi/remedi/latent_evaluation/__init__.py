"""Descriptor-space analysis: what the latent space *looks like*.

Capacity diagnostics, distributions, UMAP/PCA projections, HDBSCAN clusters read
against tmQM chemistry (metal, donor sets, geometry), chemiscope exports and the
MACE invariant statistics. This is intrinsic analysis of a descriptor matrix, not
prediction evaluation, so it is not part of ``remedi_eval``'s probe panel.

:class:`~remedi.latent_evaluation.framework_task.DescriptorAnalysisConfig`
(``kind: descriptor_analysis``) is the one entry point: a task of the
``remedi.embedding`` manifest, so ``scripts/evaluation/embed_and_evaluate.py``
embeds the dataset with a checkpoint and runs the analyses in the same job.
"""

from .analysis_tasks import (
    CapacityDiagnosticTask,
    ChemiscopeClusterTask,
    ChemiscopeProjectionTask,
    ClusterAxisAnalysisTask,
    ClusterChemicalFingerprintTask,
    ClusterGranularitySweepTask,
    DescriptorAnalysisTask,
    DescriptorDistributionTask,
    DescriptorStructureBenchmarkTask,
    HDBSCANClusterTask,
    ProjectionPlotTask,
    TopNormDescriptorsTask,
)
from .coloring import (
    ColorProvider,
    CoordinationNumberColor,
    DBlockColor,
    MetalCenterAtomicNumberColor,
    MetalCenterElementColor,
    NumAtomsColor,
    RegressionTargetColor,
)
from .context import (
    DescriptorAnalysisContext,
    DescriptorNormalizationConfig,
)

__all__ = [
    "CapacityDiagnosticTask",
    "ChemiscopeClusterTask",
    "ChemiscopeProjectionTask",
    "ClusterAxisAnalysisTask",
    "ClusterChemicalFingerprintTask",
    "ClusterGranularitySweepTask",
    "ColorProvider",
    "CoordinationNumberColor",
    "DBlockColor",
    "DescriptorAnalysisContext",
    "DescriptorAnalysisTask",
    "DescriptorDistributionTask",
    "DescriptorNormalizationConfig",
    "DescriptorStructureBenchmarkTask",
    "HDBSCANClusterTask",
    "MetalCenterAtomicNumberColor",
    "MetalCenterElementColor",
    "NumAtomsColor",
    "ProjectionPlotTask",
    "RegressionTargetColor",
    "TopNormDescriptorsTask",
]
