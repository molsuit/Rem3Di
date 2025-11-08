
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline

from threedscriptors.evaluation.clustering import (
    plot_reduced_dimension,
)
from threedscriptors.evaluation.clustering.tmqm_clustering_utils import (
    get_atomic_num_colors,
    get_block_colors,
    get_coordination_numbers,
    get_metal_center_type,
    get_tm_colormap,
)
from threedscriptors.evaluation.evaluation_utils import (
    evaluate_molecular_descriptor_on_dataset,
)
from threedscriptors.model.model_builder import ModelBuilder

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/159-2025_08_19_14_29_24-tmqmpretrained"


#model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/144-2025_07_30_14_47_16-TMQM First Run"

model = ModelBuilder.from_directory(model_directory).build_remedi_model()

dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/tmqm"
)
out_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/tmqm_pretraining/train"

dataset = reload_dataset_pipeline(dataset_directory).build()

import numpy as np

descriptors = evaluate_molecular_descriptor_on_dataset(model, dataset)
descriptors = descriptors.numpy()


# Z-score normalization: subtract mean and divide by std for each feature
descriptors = (descriptors - np.mean(descriptors, axis=0)) / np.std(descriptors, axis=0)

descriptors = descriptors / np.linalg.norm(descriptors, axis= 1, keepdims =True)


import umap

um = umap.UMAP()

emb = um.fit_transform(descriptors)





num_atoms = [len(m) for m in dataset.molecules]
fig = plot_reduced_dimension(emb, color = num_atoms, suptitle="By Number of atoms")
fig.savefig(f"{out_dir}/number_of_atoms.png", dpi = 300)


atomic_num = get_metal_center_type(dataset.molecules)
element_colors, handles = get_atomic_num_colors(atomic_num)
fig = plot_reduced_dimension(emb, color = element_colors, suptitle="By metal center", handles=handles)
fig.savefig(f"{out_dir}/metal_center_element.png", dpi = 300)

block_colors = get_block_colors(atomic_num)
fig = plot_reduced_dimension(emb, color = dataset.regression_targets[:,0], suptitle="By homo_lumo_gap")
fig.savefig(f"{out_dir}/umap_homo_lumo_gap.png", dpi = 300)





tm_cmap, norm = get_tm_colormap()

fig = plot_reduced_dimension(emb, color =atomic_num, suptitle="By metal center", cmap = tm_cmap, norm = norm)

fig.savefig(f"{out_dir}/metal_center_dblock.png", dpi = 300)




cns = get_coordination_numbers(dataset.molecules)

fig = plot_reduced_dimension(emb, color =cns)
fig.savefig(f"{out_dir}/coordination_number.png", dpi = 300)
