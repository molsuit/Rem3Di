

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis import (
    ClusteringCalculator,
    plot_reduced_dimension,
)
from threedscriptors.evaluation.evaluation_pipeline import BaseEvalTask
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
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

        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)

        self.reduced_dimensions = (
            self.clustering_calculator.get_dimensionality_reduction(self.descriptors, k=3)
        )


    def plot(self):
        assert self.reduced_dimensions is not None
        figs = {}
        figs["Descriptor_UMAP"] = plot_reduced_dimension(self.reduced_dimensions)

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

        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)

        self.reduced_dimensions = (
            self.clustering_calculator.get_dimensionality_reduction(self.descriptors, k=3)
        )

    def plot(self):
        assert self.reduced_dimensions is not None
        figs = {}
        figs["Descriptor_UMAP"] = plot_reduced_dimension(self.reduced_dimensions)


        self.figs = figs


class DescriptorElementAnalysis(BaseEvalTask):

    def __init__(self, dataset):
        super().__init__()

        self.dataset = dataset

    def run(self, model: MultiTaskRegressionModel):
        self.descriptors = evaluate_molecular_descriptor_on_dataset(model, self.dataset)

        self.descriptor_norms = torch.norm(self.descriptors, dim=1)

        self.run_capacity_diagnostics()

    def run_capacity_diagnostics(self):

        H_tot, utilisation, dead_dims, eig, d_eff = capacity_diagnostics(
            self.descriptors
        )

        self.eig = eig
        self.results.update(
            {
                "Largest 10 Eigenvalues": eig[::-1][:10].tolist(),
                "Smallest 10 Eigenvalues": eig[:10].tolist(),
                "Effective dimension": d_eff.item(),
                "Total entropy": H_tot.item(),
                "Utilisation": utilisation.item(),
                "Dead dims": dead_dims,
            }
        )

    def plot(self):
        fig_hist = plt.figure()
        plt.hist(
            self.descriptors.reshape(-1),
            bins=np.linspace(
                torch.min(self.descriptors), torch.max(self.descriptors), 50
            ),
        )
        plt.yscale("log")

        fig_eigval = plt.figure()
        plt.semilogy(self.eig)
        plt.xlabel("#")
        plt.ylabel("eigenvalue")

        fig_discriptor_norm = plt.figure()
        plt.hist(
            self.descriptor_norms,
            bins=np.linspace(
                torch.min(self.descriptor_norms), torch.max(self.descriptor_norms), 50
            ),
        )
        plt.yscale("log")
        plt.title("Distribution of Descriptor L2 Norms")

        self.figs = {
            "DescriptorElementDistribution": fig_hist,
            "EigenvalueHistogram": fig_eigval,
            "DescriptorNorm": fig_discriptor_norm,
        }
