from matplotlib.pyplot import Figure

from threedscriptors.data_handling.dataset import AtomicEmbeddingDataset
from threedscriptors.data_handling.dataset_io import load_data_from_disk
from threedscriptors.evaluation.clustering import UMAPCalculator, PCACalculator
from threedscriptors.evaluation.evaluation_pipeline import EnolThiolEvalTask
from threedscriptors.model.model_builder import ModelBuilder

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/cmrt"
model = ModelBuilder.from_directory(model_directory).build_model()

dataset_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/functional_group_dataset"
dataset = load_data_from_disk(dataset_directory, dataset_cls=AtomicEmbeddingDataset)

task = EnolThiolEvalTask(dataset=dataset, clustering_calculator=PCACalculator())


task.run(model)
plotting_output = task.plot()

fig: Figure = plotting_output["FunctionalGroupComparisonTask"]

fig.savefig("Functional_group_dim_reduction.pdf")
