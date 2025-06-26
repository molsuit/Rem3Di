from abc import ABC, abstractmethod
from threedscriptors.evaluation.evaluation_pipeline import DescriptorPCATask, RegressionTestTask, EvalPipelineRunner, SimilarityScreeningTask, DescriptorSimilarityAnalysisTask, RegressionHeadPCATask, ChiralPredictionTask, PreprocessorVisualizationTask, DescriptorElementAnalysis

import matplotlib.pyplot as plt

from threedscriptors.data_handling.dataset import SimilarityScreeningDataset, AtomicEmbeddingDataset

from threedscriptors.evaluation.clustering import (
    ClusteringCalculator,
    UMAPCalculator,
    PCACalculator,
    plot_reduced_dimension,
    plot_reduced_dimension_functional_group_comparison,
)
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline

from threedscriptors.evaluation.descriptor_calculators import (
    MolfeatDescriptorCalculator,
    ThreedescriptorCalculator,
)

from threedscriptors.evaluation.similarity_screening import (
    SimilarityMetrics,
    SimilarityScreening,
    plot_reference_vs_model_classification_metric,
    plot_roc
)
from threedscriptors.model.regression_models import MultiTaskRegressionModel


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