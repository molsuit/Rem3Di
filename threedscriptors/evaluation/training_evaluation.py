from threedscriptors.evaluation.clustering import (
    PCACalculator,
    UMAPCalculator,
)
from threedscriptors.evaluation.evaluation_pipeline import (
    ChiralPredictionTask,
    DescriptorElementAnalysis,
    DescriptorPCATask,
    EvalPipelineRunner,
    PreprocessorVisualizationTask,
    RegressionTestTask,
)

from threedscriptors.data_handling.dataset import BaseDataset


def regression_pipeline(dataset: BaseDataset):

    tasks = [
        RegressionTestTask(dataset),
        DescriptorPCATask(dataset, UMAPCalculator()),
        DescriptorPCATask(dataset, PCACalculator()),
        DescriptorElementAnalysis(dataset),
        #PreprocessorVisualizationTask(dataset),
        # RegressionHeadPCATask(dataset, UMAPCalculator()),
        # SimilarityScreeningTask(SIMILARITY_SCREENING_DATASET),
        # DescriptorSimilarityAnalysisTask(train_dataset),
    ]

    return EvalPipelineRunner(
        tasks=tasks,
        dataset_name=dataset.dataset_config.dataset_name,
        dataset_split=dataset.dataset_config.dataset_split,
    )


def chiral_regression_pipeline(dataset):

    tasks = [ChiralPredictionTask(dataset), PreprocessorVisualizationTask(dataset)]

    return EvalPipelineRunner(tasks=tasks)
