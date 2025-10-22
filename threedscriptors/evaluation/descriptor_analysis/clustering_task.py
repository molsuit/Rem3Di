import chemiscope
import matplotlib.pyplot as plt
import numpy as np
import torch

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.evaluation.descriptor_analysis import (
    ClusteringCalculator,
    plot_reduced_dimension,
)
from threedscriptors.evaluation.descriptor_analysis.capacity_diagnostic import (
    get_descriptor_channel_distribution,
    get_descriptor_norm_distribution,
    run_latent_space_capacity_diagnostic,
)
from threedscriptors.evaluation.evaluation_pipeline import BaseEvalTask
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from threedscriptors.evaluation.results import (
    ChemiscopeResult,
    FigureResult,
    PydanticResult,
)
from threedscriptors.model.regression_models import MultiTaskRegressionModel


class DescriptorClusteringTask(BaseEvalTask):
    "Plots the PCA results of the Molecular Descriptor"

    def __init__(
        self, dataset: MoleculeDataset, clustering_calculator: ClusteringCalculator
    ):
        super().__init__()
        self.dataset = dataset
        self.clustering_calculator = clustering_calculator
        self.reduced_dimensions = None

    def run(self, model: MultiTaskRegressionModel):
        # evaluate model to get descriptors

        train_dataset = TrainingMoleculeDataset.from_molecule_dataset(self.dataset, get_item=pos_emb_getitem)

        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, train_dataset)

        self.reduced_dimensions = (
            self.clustering_calculator.get_dimensionality_reduction(self.descriptors, k=2)
        )

    def plot(self,model_name: str):
        assert self.reduced_dimensions is not None

        fig = plot_reduced_dimension(self.reduced_dimensions)

        self.results.append(FigureResult(file_name=f"umap_{model_name}.png", figure = fig))

        chemiscope_input = self._get_chemiscope_umap()

        self.results.append(ChemiscopeResult(file_name= f"umap_{model_name}.json.gz", data = chemiscope_input))

    def _get_chemiscope_umap(self):
        chemiscope_properties = {"PC" : self.reduced_dimensions}
        chemiscope_settings = chemiscope.quick_settings(x="PC[1]",y="PC[2]")
        frames = self.dataset.get_all_molecules()
        return chemiscope.create_input(frames=frames, properties=chemiscope_properties, settings = chemiscope_settings)


class DescriptorElementAnalysis(BaseEvalTask):

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel):

        train_dataset = TrainingMoleculeDataset.from_molecule_dataset(self.dataset, get_item=pos_emb_getitem)

        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, train_dataset)

        self.capacity_diagnostic_result = run_latent_space_capacity_diagnostic(
            self.descriptors
        )
        self.results.append(PydanticResult(file_name="capacity_diagnostic.yaml",obj = self.capacity_diagnostic_result))

        self.mean, self.std = get_descriptor_channel_distribution(self.descriptors)

        self.descriptor_norms = get_descriptor_norm_distribution(self.descriptors)



    def plot(self, model_name: str):

        fig_hist = plt.figure()
        plt.hist(
            self.descriptors.reshape(-1),
            bins=np.linspace(
                torch.min(self.descriptors), torch.max(self.descriptors), 50
            ),
        )
        plt.yscale("log")
        self.results.append(FigureResult(file_name=f"{model_name}_descriptor_element_distribution.png", figure = fig_hist))


        fig_discriptor_norm = plt.figure()
        plt.hist(
            self.descriptor_norms,
            bins=np.linspace(
                np.min(self.descriptor_norms), np.max(self.descriptor_norms), 50
            ),
        )
        plt.yscale("log")
        plt.title("Distribution of Descriptor L2 Norms")
        self.results.append(FigureResult(file_name=f"{model_name}_descriptor_norm_distribution.png", figure = fig_discriptor_norm))

