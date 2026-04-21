import torch
from matplotlib.pyplot import Figure

from threedscriptors.configuration.architecture_config import (
    EncoderOnlyArchitectureConfig,
)
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.clustering import PCACalculator
from threedscriptors.evaluation.evaluation_pipeline import EnolThiolEvalTask

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/139-2025_07_30_10_44_46-400kpcqm_hot_with_l1"
model = EncoderOnlyArchitectureConfig.from_directory(model_directory).build()
model.encoder.load_state_dict(torch.load(f"{model_directory}/encoder.pth"))
model.preprocessor.atomic_preprocessor.load_state_dict(
    torch.load(f"{model_directory}/atomic_preprocessor.pth")
)
model.preprocessor.geometric_preprocessor.load_state_dict(
    torch.load(f"{model_directory}/geometric_preprocessor.pth")
)

dataset_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/functional_group_dataset"


dataset = reload_dataset_pipeline(dataset_directory).build()


task = EnolThiolEvalTask(dataset=dataset, clustering_calculator=PCACalculator())


task.run(model)
plotting_output = task.plot()

fig: Figure = plotting_output["FunctionalGroupComparisonTask"]

fig.savefig("Functional_group_dim_reduction.pdf")
