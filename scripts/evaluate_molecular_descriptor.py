from matplotlib.pyplot import Figure

from threedscriptors.data_handling.dataset import AtomicEmbeddingDataset
from threedscriptors.data_handling.dataset_io import load_data_from_disk
from threedscriptors.evaluation.clustering import UMAPCalculator
from threedscriptors.evaluation.evaluation_pipeline import EnolThiolEvalTask
from threedscriptors.model.model_builder import ModelBuilder

model_directory = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/cmrt_ps"
model = ModelBuilder.from_directory(model_directory).build_model()

dataset_directory = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/functional_group_dataset"
dataset = load_data_from_disk(dataset_directory, dataset_cls=AtomicEmbeddingDataset)

task = EnolThiolEvalTask(dataset=dataset, clustering_calculator=UMAPCalculator())


task.run(model)
plotting_output = task.plot()

fig: Figure = plotting_output["FunctionalGroupComparisonTask"]

fig.savefig("Functional_group_dim_reduction.pdf")
