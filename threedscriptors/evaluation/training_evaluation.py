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

#SIMILARITY_SCREENING_DATASET_PATH = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/virtual_screening"
#SIMILARITY_SCREENING_DATASET = reload_dataset_pipeline(SIMILARITY_SCREENING_DATASET_PATH, normalize_targets= False, dataset_cls=AtomicEmbeddingDataset).build()


def regression_pipeline(train_dataset, valid_dataset):

    tasks = [RegressionTestTask(valid_dataset),
            DescriptorPCATask(train_dataset, UMAPCalculator()),
            DescriptorPCATask(train_dataset, PCACalculator()),
            DescriptorElementAnalysis(train_dataset),
            DescriptorElementAnalysis(valid_dataset),
            PreprocessorVisualizationTask(train_dataset)
            #RegressionHeadPCATask(dataset, UMAPCalculator()),
            #SimilarityScreeningTask(SIMILARITY_SCREENING_DATASET),
            #DescriptorSimilarityAnalysisTask(train_dataset),
        ]

    return EvalPipelineRunner(tasks = tasks)


def chiral_regression_pipeline(dataset):

    tasks = [
        ChiralPredictionTask(dataset),
        PreprocessorVisualizationTask(dataset)
    ]

    return EvalPipelineRunner(tasks = tasks)
