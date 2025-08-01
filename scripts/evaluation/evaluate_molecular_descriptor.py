from matplotlib.pyplot import Figure

from threedscriptors.evaluation.clustering import PCACalculator, UMAPCalculator
from threedscriptors.evaluation.evaluation_pipeline import EnolThiolEvalTask
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/139-2025_07_30_10_44_46-400kpcqm_hot_with_l1"
model = ModelBuilder.from_directory(model_directory).build_remedi_model()

dataset_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/functional_group_dataset"


dataset = reload_dataset_pipeline(dataset_directory).build()



task = EnolThiolEvalTask(dataset=dataset, clustering_calculator=PCACalculator())


task.run(model)
plotting_output = task.plot()

fig: Figure = plotting_output["FunctionalGroupComparisonTask"]

fig.savefig("Functional_group_dim_reduction.pdf")
