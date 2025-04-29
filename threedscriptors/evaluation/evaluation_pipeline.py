from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from threedscriptors.data_handling.dataset import (
    BaseDataset,
    RegressionDataset,
    RegressionWithAuxDataset,
    SimilarityScreeningDataset,
)
from threedscriptors.evaluation.clustering import (
    ClusteringCalculator,
    plot_reduced_dimension,
    plot_reduced_dimension_functional_group_comparison,
)
from threedscriptors.evaluation.descriptor_calculators import (
    MolfeatDescriptorCalculator,
    ThreedescriptorCalculator,
)
from threedscriptors.evaluation.descriptor_similarity_metrics import (
    cosine_similarity_matrix,
    euclidean_similarity,
    plot_similarity_matrix,
)
from threedscriptors.evaluation.eval_utils import (
    evaluate_molecular_descriptor_on_dataset,
    evaluate_regression_model_on_dataset,
)
from threedscriptors.evaluation.regression_analysis import (
    add_regression_head_activations_hooks,
)
from threedscriptors.evaluation.similarity_screening import (
    SimilarityMetrics,
    SimilarityScreening,
    plot_reference_vs_model_classification_metric,
)
from threedscriptors.model.regression_models import MultiTaskRegressionModel


class BaseEvalTask(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def run(self, model: MultiTaskRegressionModel):
        pass

    @abstractmethod
    def plot(self, model_name: str):
        pass


class DescriptorPCATask(BaseEvalTask):
    "Plots the PCA results of the Molecular Descriptor"

    def __init__(
        self, dataset: BaseDataset, clustering_calculator: ClusteringCalculator
    ):
        self.dataset = dataset
        self.clustering_calculator = clustering_calculator
        self.reduced_dimensions = None

    def run(self, model: MultiTaskRegressionModel):
        # evaluate model to get descriptors

        descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)

        self.reduced_dimensions = (
            self.clustering_calculator.get_dimensionality_reduction(descriptors)
        )

    def plot(self):
        assert self.reduced_dimensions is not None
        figs = {}
        figs["DescriptorPCA"] = plot_reduced_dimension(self.reduced_dimensions)

        return figs


class RegressionHeadPCATask(BaseEvalTask):
    "Run the PCA analysis for the activations in each head"

    def __init__(self, dataset, clustering_calculator: ClusteringCalculator):
        self.dataset = dataset
        self.clustering_calculator = clustering_calculator

    def run(self, model: MultiTaskRegressionModel):
        activations = add_regression_head_activations_hooks(model)
        evaluate_regression_model_on_dataset(model, self.dataset)

        self.activations_dim_reduced = {}

        for task_head_layer, activation in activations.items():
            self.activations_dim_reduced[task_head_layer] = (
                self.clustering_calculator.get_dimensionality_reduction(activation, k=2)
            )

    def plot():
        pass


class RegressionTestTask(BaseEvalTask):
    def __init__(self, dataset: RegressionDataset | RegressionWithAuxDataset):
        self.dataset = dataset

    def run(self, model):
        assert set(
            [tc.task_name for tc in self.dataset.dataset_config.tasks]
        ).issubset(
            set(model.multitask_heads.task_list)
        )  # Check that tasks in the dataset are actually present in the prediction heads.

    def _check_uncertainty(self):
        "Plots the uncertainty of the regression predictions within the conformers of a single molecule"

    def plot(self):
        figs = {}

        figs.update(self._plot_prediction_vs_reference())

    def _plot_prediction_vs_reference(self) -> dict[str : plt.Figure]:
        pass


class SimilarityScreeningTask(BaseEvalTask):
    def __init__(self, dataset):
        self.dataset: SimilarityScreeningDataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        threedscriptor = ThreedescriptorCalculator(
            model, similarity_fn=euclidean_similarity
        )
        self.model_screening = SimilarityScreening(threedscriptor, self.dataset)
        self.model_screening.evaluate(actives_resampling_frequency=10)
        self.model_screening.compute_metrics(enrichment_factor_percentage=0.01)

        ecfp = MolfeatDescriptorCalculator(descriptor_name="ecfp")
        self.reference_screening = SimilarityScreening(ecfp, self.dataset)
        self.reference_screening.evaluate(actives_resampling_frequency=10)
        self.reference_screening.compute_metrics(enrichment_factor_percentage=0.01)

    def plot(self):
        metrics = [SimilarityMetrics.AUROC, SimilarityMetrics.ENRICHMENT_FACTOR]
        figs = {}

        for metric in metrics:
            figs[f"SimilarityScreening{metric!s}"] = (
                plot_reference_vs_model_classification_metric(
                    threedscriptor_screening=self.model_screening,
                    reference_screening=self.reference_screening,
                    metric=metric,
                )
            )

        return figs


class ChiralPredictionTask(BaseEvalTask): ...


class DescriptorSimilarityAnalysisTask(BaseEvalTask):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)
        self.similarity_matrix = cosine_similarity_matrix(self.descriptors)

    def plot(self):
        figs = {}

        figs["CosineSimilarityMatrixOfMolDescriptors"] = plot_similarity_matrix(
            self.similarity_matrix
        )


class EnolThiolEvalTask(DescriptorPCATask):
    def __init__(self, dataset: BaseDataset, clustering_calculator):
        super().__init__(dataset=dataset, clustering_calculator=clustering_calculator)

    def plot(self) -> plt.Figure:
        fig = plot_reduced_dimension_functional_group_comparison(
            self.reduced_dimensions, self.dataset.smiles_list
        )
        return {"FunctionalGroupComparisonTask": fig}


class EvalPipelineRunner:
    def __init__(self, tasks: list[BaseEvalTask]):
        self.tasks = tasks

    def evaluate(self, model: MultiTaskRegressionModel):
        model.eval()

        for task in self.tasks:
            task.run(model)

    def visualize(self, output_directory: str, model_name: str):
        figs: dict[str : plt.Figure] = {}  # taskname : Figure
        for task in self.tasks:
            figs.update(task.plot())

        for fig_name, fig in figs.items():
            fig.savefig(f"{output_directory}/{fig_name}_{model_name}.pdf")
