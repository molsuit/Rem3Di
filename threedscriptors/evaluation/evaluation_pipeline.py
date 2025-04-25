from abc import ABC, abstractmethod
from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from threedscriptors.data_handling.dataset import (
    BaseDataset,
    RegressionDataset,
    RegressionWithAuxDataset,
)
from threedscriptors.data_handling.sample import Sample, sample_collate_fn
from threedscriptors.evaluation.descriptor_similarity_metrics import (
    cosine_similarity_matrix,
    plot_similarity_matrix,
)
from threedscriptors.evaluation.regression_analysis import (
    get_regression_head_activations,
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

    ...


class RegressionHeadPCATask(BaseEvalTask):
    "Run the PCA analysis for the activations in each head"

    def __init__(self, dataset):
        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        activations = get_regression_head_activations(model, self.dataset)

        return activations


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


class SimilarityScreeningTask(BaseEvalTask): ...


class ChiralPredictionTask(BaseEvalTask): ...


class DescriptorSimilarityAnalysisTask(BaseEvalTask):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        batch_size = 128
        dataloader: Iterable[Sample] = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=sample_collate_fn,
        )

        device = "cuda"

        descriptors = torch.zeros(
            size=(len(self.dataset.mol_ids), model.global_aggregator.config.output_dim)
        )

        for batch_idx, samples in enumerate(dataloader):
            embeddings = samples.embeddings.to(device)
            padding_mask = samples.padding_mask.to(device)

            descriptors[batch_idx * batch_size : (batch_idx + 1) * batch_size] = (
                model.get_molecular_descriptor(embeddings, padding_mask)
            )

        self.descriptors = descriptors

        self.similarity_matrix = cosine_similarity_matrix(self.descriptors)

    def plot(self):
        figs = {}

        figs["CosineSimilarityMatrixOfMolDescriptors"] = plot_similarity_matrix(
            self.similarity_matrix
        )


class EnolThiolEvalTask(DescriptorPCATask):
    def __init__(self, dataset: BaseDataset):
        pass

    def plot(self) -> plt.Figure:
        pass

    # Calculate descriptors


class EvalPipelineRunner:
    def __init__(self, tasks: list[BaseEvalTask]):
        self.tasks = tasks

    def evaluate(self, model: MultiTaskRegressionModel):
        model.eval()

        for task in self.tasks:
            task.run()

    def visualize(self, output_directory: str, model_name: str):
        figs: dict[str : plt.Figure] = {}  # taskname : Figure
        for task in self.tasks:
            figs.update(task.plot())

        for fig_name, fig in figs.items():
            fig.savefig(f"{output_directory}/{fig_name}_{model_name}.pdf")
