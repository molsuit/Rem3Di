"""Descriptor-space analysis: what a descriptor's space *looks like*.

Capacity diagnostics, distributions, UMAP/PCA projections, HDBSCAN clusters read
against tmQM chemistry (metal, donor sets, geometry) and chemiscope exports.
This is intrinsic analysis of a descriptor matrix, not prediction evaluation.

:class:`DescriptorAnalysisConfig` (``kind: descriptor_analysis``) is the one
entry point: a task of the eval manifest that analyses the run's ``model``
descriptor (ECFP, precomputed embeddings, a Rem3Di checkpoint's embeddings)
through :meth:`EvalContext.descriptor_matrix`. The tmQM chemistry (pymatgen)
and the chemiscope exports need the ``descriptor-analysis`` extra.
"""

from remedi_eval.framework.tasks.descriptor_analysis.analysis_tasks import (
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
from remedi_eval.framework.tasks.descriptor_analysis.coloring import (
    ColorProvider,
    CoordinationNumberColor,
    DBlockColor,
    MetalCenterAtomicNumberColor,
    MetalCenterElementColor,
    NumAtomsColor,
    RegressionTargetColor,
)
from remedi_eval.framework.tasks.descriptor_analysis.context import (
    DescriptorAnalysisContext,
    DescriptorNormalizationConfig,
)
from remedi_eval.framework.tasks.descriptor_analysis.task import (
    DescriptorAnalysisConfig,
    DescriptorAnalysisError,
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
    "DescriptorAnalysisConfig",
    "DescriptorAnalysisContext",
    "DescriptorAnalysisError",
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
