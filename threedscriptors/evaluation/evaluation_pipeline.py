from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import torch

from threedscriptors.evaluation.analysis import compute_class_std
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
    plot_roc
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

        descriptors = evaluate_molecular_descriptor_on_dataset(
            model, self.dataset)

        self.reduced_dimensions = (
            self.clustering_calculator.get_dimensionality_reduction(
                descriptors)
        )

    def plot(self):
        assert self.reduced_dimensions is not None
        figs = {}
        figs[f"Descriptor_{type(self.clustering_calculator)}"] = plot_reduced_dimension(self.reduced_dimensions)

        return figs


class RegressionHeadPCATask(BaseEvalTask):
    "Run the PCA analysis for the activations in each head"

    def __init__(self, dataset, clustering_calculator: ClusteringCalculator):
        self.dataset = dataset
        self.clustering_calculator = clustering_calculator

    def run(self, model: MultiTaskRegressionModel):
        activations = add_regression_head_activations_hooks(model)
        evaluate_regression_model_on_dataset(model, self.dataset)
        
        print(activations.keys())

        breakpoint()

        for k, v in activations.items(): 
            print(k)
            print(len(v))
            for t in v:
                print(t.shape)

        self.activations_dim_reduced = {}

        for task_head_layer, activation in activations.items():
            self.activations_dim_reduced[task_head_layer] = (
                self.clustering_calculator.get_dimensionality_reduction(
                    activation, k=2)
            )

    def plot(self):
        pass


class RegressionTestTask(BaseEvalTask):
    def __init__(self, dataset: RegressionDataset | RegressionWithAuxDataset):
        self.dataset = dataset
        self.labels = self.dataset.regression_targets

    def run(self, model):
        assert set(
            [tc.task_name for tc in self.dataset.dataset_config.tasks]
        ).issubset(
            set(model.multitask_heads.task_list)
            # Check that tasks in the dataset are actually present in the prediction heads.
        )

        model.to("cuda")

        self.predictions = evaluate_regression_model_on_dataset(
            model, self.dataset)

        self._check_uncertainty()

    def _check_uncertainty(self):
        "Plots the uncertainty of the regression predictions within the conformers of a single molecule"

        labeled_std_devs = []
        unlabeld_std_devs = []

        for task_idx, task in enumerate(self.dataset.dataset_config.tasks):

            mol_ids_with_labels, predictions_with_labels, labels = self.get_labeled_task_data(
                task_idx)

            # This calculates the std deviation for the predicitions that are labeled and not masked
            _, task_conformer_std_dev_labeled = compute_class_std(
                predictions_with_labels.reshape(-1,1), mol_ids_with_labels)

            mol_ids_without_labels, predictions_without_lables = self.get_unlabeled_task_data(
                task_idx)

            # This calculates the std deviation for the predicitions that are unlabeld
            _, task_conformer_std_dev_unlabeled = compute_class_std(
                predictions_without_lables.reshape(-1,1), mol_ids_without_labels)


            labeled_std_devs.append(task_conformer_std_dev_labeled)
            unlabeld_std_devs.append(task_conformer_std_dev_unlabeled)

        self.results = {"labeld_std_devs_conf_predictions" : labeled_std_devs,
                        "unlabeled_std_devs_conf_predictions" : unlabeld_std_devs} 

        print(self.results)
            

    def get_unlabeled_task_data(self, task_idx):

        task_predictions = self.predictions[:, task_idx]

        unlabeld_task_mask = torch.logical_not(torch.tensor(
            self.dataset.regression_masks[:, task_idx], dtype=bool))

        predictions_without_lables = task_predictions[unlabeld_task_mask]
        mol_ids_without_labels =  [
            mol_id
            for mol_id, keep in zip(self.dataset.mol_ids, unlabeld_task_mask)
            if keep
        ]

        return mol_ids_without_labels, predictions_without_lables

    def get_labeled_task_data(self, task_idx):
        task_mask = torch.tensor(
            self.dataset.regression_masks[:, task_idx], dtype=bool)

        task_predictions = self.predictions[:, task_idx]
        predictions_with_labels = task_predictions[task_mask].detach(
        ).cpu().numpy()

        task_labels = self.dataset.regression_targets[:, task_idx]
        labels = task_labels[task_mask].detach(
        ).cpu().numpy()

        mol_ids_with_labels = [
            mol_id
            for mol_id, keep in zip(self.dataset.mol_ids, task_mask)
            if keep
        ]

        return mol_ids_with_labels, predictions_with_labels, labels

    def plot(self):
        figs = {}

        figs.update(self._plot_prediction_vs_reference())
        figs.update(self._plot_conformer_uncertainty_histogram())
        return figs
    
    def _plot_conformer_uncertainty_histogram(self):
        unlabeled_std_dev = np.squeeze(np.concatenate(self.results
        ["unlabeled_std_devs_conf_predictions"]))

        labeled_std_dev = np.squeeze(np.concatenate(self.results["labeld_std_devs_conf_predictions"]))


        fig = plt.figure(figsize = (8,6))

        max_std_dev = max(np.max(unlabeled_std_dev), np.max(labeled_std_dev))
        min_std_dev = min(np.min(unlabeled_std_dev), np.min(labeled_std_dev))



        
        bins= np.logspace(np.log10(min_std_dev), np.log10(max_std_dev), 25)
        bins = np.linspace(min_std_dev, 0.5, 50)
        labeled_counts, labeled_bins = np.histogram(labeled_std_dev, bins= bins)
        unlabeled_counts, unlabeled_bins = np.histogram(unlabeled_std_dev, bins = bins)


        plt.stairs(labeled_counts, labeled_bins, label= "Labelled Predictions")
        plt.stairs(unlabeled_counts, unlabeled_bins, label= "Unlabelled Predictions")


        plt.ylabel("Counts")
        plt.xlabel("Std deviation of Model Predictions between conformers of the same molecule (in std. units)")
        #plt.xscale("log")
        #plt.yscale("log")
        plt.legend()

        return {"Conformer_std_dev" : fig}

    def _plot_prediction_vs_reference(self) -> dict[str: plt.Figure]:
        fig, ax = plt.subplots()
        for task_idx, task in enumerate(self.dataset.dataset_config.tasks):

            _, predictions_with_labels, labels = self.get_labeled_task_data(
                task_idx)
            plt.scatter(predictions_with_labels, labels, label=task.task_name, alpha = 0.6, s= 0.5)


        plt.xlabel("Predictions")
        plt.ylabel("Reference Labels")

        ax.legend(
        bbox_to_anchor=(1.07, 1),  # x=1.02 means just to the right of the axes
        loc='upper left',
        borderaxespad=0
    )
        #fig.tight_layout(rect=[0, 0, 0.85, 1])


        return {"RefVSPredScatter": fig}


class SimilarityScreeningTask(BaseEvalTask):
    def __init__(self, dataset):
        self.dataset: SimilarityScreeningDataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        threedscriptor = ThreedescriptorCalculator(
            model, similarity_fn=euclidean_similarity
        )
        self.model_screening = SimilarityScreening(
            threedscriptor, self.dataset)
        self.model_screening.evaluate(actives_resampling_frequency=10)
        self.model_screening.compute_metrics(enrichment_factor_percentage=0.01)
        print(self.model_screening.results)

        ecfp = MolfeatDescriptorCalculator(descriptor_name="ecfp")
        self.reference_screening = SimilarityScreening(ecfp, self.dataset)
        self.reference_screening.evaluate(actives_resampling_frequency=10)
        self.reference_screening.compute_metrics(
            enrichment_factor_percentage=0.01)
        print(self.reference_screening.results)
        

    def plot(self):
        metrics = [SimilarityMetrics.AUROC,
                   SimilarityMetrics.ENRICHMENT_FACTOR]
        figs = {}

        for metric in metrics:
            figs[f"SimilarityScreening{metric!s}"] = (
                plot_reference_vs_model_classification_metric(
                    threedscriptor_screening=self.model_screening,
                    reference_screening=self.reference_screening,
                    metric=metric,
                )
            )

        figs["ROCs"] = plot_roc(self.model_screening,
                                 self.reference_screening)

        return figs


class ChiralPredictionTask(BaseEvalTask):
    ...


class DescriptorSimilarityAnalysisTask(BaseEvalTask):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        self.descriptors = evaluate_molecular_descriptor_on_dataset(
            model, self.dataset)
        self.similarity_matrix = cosine_similarity_matrix(self.descriptors)

    def plot(self):
        figs = {}

        figs["CosineSimilarityMatrixOfMolDescriptors"] = plot_similarity_matrix(
            self.similarity_matrix
        )

        return figs


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
        figs: dict[str: plt.Figure] = {}  # taskname : Figure
        for task in self.tasks:
            figs.update(task.plot())

        for fig_name, fig in figs.items():
            fig.savefig(f"{output_directory}/{fig_name}_{model_name}.pdf")
