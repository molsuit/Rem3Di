from threedscriptors.evaluation.clustering import (
    PCACalculator,
    UMAPCalculator,
)
from threedscriptors.evaluation.evaluation_pipeline import (
    ChiralPredictionTask,
    DescriptorElementAnalysis,
    DescriptorClusteringTask,
    EvalPipelineRunner,
    RegressionTestTask,
)

from threedscriptors.configuration.data_config import DatasetSplit

from threedscriptors.data_handling.dataset import BaseDataset


def regression_pipeline(dataset: BaseDataset, dataset_split: DatasetSplit | None = None):

    if dataset_split is None:
        dataset_split = dataset.dataset_config.dataset_split

    tasks = [
        RegressionTestTask(dataset),
        DescriptorClusteringTask(dataset, UMAPCalculator()),
        DescriptorElementAnalysis(dataset),
        # PreprocessorVisualizationTask(dataset),
        # RegressionHeadPCATask(dataset, UMAPCalculator()),
        # SimilarityScreeningTask(SIMILARITY_SCREENING_DATASET),
        # DescriptorSimilarityAnalysisTask(train_dataset),
    ]

    return EvalPipelineRunner(
        tasks=tasks,
        dataset_name=dataset.dataset_config.dataset_name,
        dataset_split=dataset_split,
    )


def chiral_regression_pipeline(dataset : BaseDataset):

    tasks = [ChiralPredictionTask(dataset)]  # , PreprocessorVisualizationTask(dataset)]

    return EvalPipelineRunner(
        tasks=tasks,
        dataset_name=dataset.dataset_config.dataset_name,
        dataset_split=dataset.dataset_config.dataset_split,
    )
