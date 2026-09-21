"""Descriptor-space analysis — what the latent space *looks like*.

Capacity diagnostics, clustering, UMAP/PCA projections, chemiscope exports and
the MACE invariant statistics. This is intrinsic analysis of a descriptor
matrix, not prediction evaluation, so it lives outside the benchmark framework
(``remedi.evaluation``) and is not part of the paper's probe panel.

It keeps its ``EvalTask`` shape: :class:`DescriptorAnalysisConfig` (in
:mod:`remedi.latent_evaluation.framework_task`) is still a runnable framework
task, but it is deliberately **not** a member of the framework's ``TaskConfig``
union, so a manifest cannot reach it until this family is worked on again.
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
from .clustering import (
    ClusteringCalculator,
    PCACalculator,
    UMAPCalculator,
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
    ProjectionConfig,
)
from .plotting import (
    plot_reduced_dimension,
    plot_reduced_dimension_3d,
    plot_reduced_dimension_functional_group_comparison,
)
from .runner import DescriptorAnalysisRunner

__all__ = [
    "CapacityDiagnosticTask",
    "ChemiscopeClusterTask",
    "ChemiscopeProjectionTask",
    "ClusterAxisAnalysisTask",
    "ClusterChemicalFingerprintTask",
    "ClusterGranularitySweepTask",
    "ClusteringCalculator",
    "ColorProvider",
    "CoordinationNumberColor",
    "DBlockColor",
    "DescriptorAnalysisContext",
    "DescriptorAnalysisRunner",
    "DescriptorAnalysisTask",
    "DescriptorDistributionTask",
    "DescriptorNormalizationConfig",
    "DescriptorStructureBenchmarkTask",
    "HDBSCANClusterTask",
    "MetalCenterAtomicNumberColor",
    "MetalCenterElementColor",
    "NumAtomsColor",
    "PCACalculator",
    "ProjectionConfig",
    "ProjectionPlotTask",
    "RegressionTargetColor",
    "TopNormDescriptorsTask",
    "UMAPCalculator",
    "plot_reduced_dimension",
    "plot_reduced_dimension_3d",
    "plot_reduced_dimension_functional_group_comparison",
]
