import matplotlib.pyplot as plt
import numpy as np

from threedscriptors.data_handling.dataset import (
    RegressionWithAuxDataset,
)
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.descriptor_calculators import (
    ThreedescriptorCalculator,
)
from threedscriptors.evaluation.descriptor_similarity_metrics import (
    cosine_similarity_matrix,
)
from threedscriptors.evaluation.plotting import plot_similarity_matrix
from threedscriptors.model.model_builder import ModelBuilder

directory = (
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/cmrt"
)

dataset = reload_dataset_pipeline(
    directory,
    normalize_inputs=True,
    normalize_targets=True,
    dataset_cls=RegressionWithAuxDataset,
).build()


eval_name = "cmrt_ps"

model_directory = f"/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/{eval_name}"

mb = ModelBuilder.from_directory(model_directory)
threedescriptor_model = mb.build_model()
threedescriptor_model.eval().float()

threedscriptor = ThreedescriptorCalculator(threedescriptor_model)
descriptors = threedscriptor.calculate_descriptors(dataset)


mean_descriptor = np.mean(descriptors, axis=0)
std_descriptor = np.std(descriptors, axis=0)

descriptors = (descriptors - mean_descriptor) / std_descriptor


sim_matrix = cosine_similarity_matrix(descriptors)
assert np.allclose(sim_matrix, sim_matrix.T)

fig = plot_similarity_matrix(sim_matrix, cmap="inferno")
fig.savefig(f"sim_mat_{eval_name}.png")


print(descriptors.shape)

plt.savefig(f"PCA_{eval_name}.png")
