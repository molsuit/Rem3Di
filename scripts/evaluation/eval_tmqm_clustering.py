
from threedscriptors.configuration.data_config import DatasetSplit
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline
from threedscriptors.evaluation.clustering import (
    UMAPCalculator,
    plot_reduced_dimension,
    plot_reduced_dimension_3d,
)
from threedscriptors.evaluation.clustering.tmqm_clustering_utils import (
    get_atomic_num_colors,
    get_block_colors,
    get_coordination_numbers,
    get_metal_center_type,
    get_tm_colormap,
)
from threedscriptors.evaluation.evaluation_pipeline import (
    DescriptorClusteringTask,
    EvalPipelineRunner,
)
from threedscriptors.model.model_builder import ModelBuilder

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/139-2025_08_01_17_27_47-pretraintmqmwithpos_hto"


#model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/144-2025_07_30_14_47_16-TMQM First Run"

model = ModelBuilder.from_directory(model_directory).build_remedi_model()

dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/tmqm"
)
out_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/tmqm_pretraining/train"

dataset = reload_dataset_pipeline(dataset_directory).build()



clustering_calculator = UMAPCalculator()
dc = DescriptorClusteringTask(
    dataset=dataset, clustering_calculator=clustering_calculator
)


runner = EvalPipelineRunner(
    tasks=[dc], dataset_name="tmqm", dataset_split=DatasetSplit.TEST
)


runner.evaluate(model)
runner.visualize(
    output_directory=out_dir,
    model_name="tmqm_regressionmodel",
)


fig = plot_reduced_dimension(dc.reduced_dimensions, color = dataset.regression_targets[:,0], suptitle="By homo_lumo_gap")
fig.savefig(f"{out_dir}/umap_homo_lumo_gap.png", dpi = 300)


num_atoms = [len(m) for m in dataset.molecules]
fig = plot_reduced_dimension(dc.reduced_dimensions, color = num_atoms, suptitle="By Number of atoms")
fig.savefig(f"{out_dir}/number_of_atoms.png", dpi = 300)


atomic_num = get_metal_center_type(dataset.molecules)
element_colors, handles = get_atomic_num_colors(atomic_num)
fig = plot_reduced_dimension(dc.reduced_dimensions, color = element_colors, suptitle="By metal center", handles=handles)
fig.savefig(f"{out_dir}/metal_center_element.png", dpi = 300)




block_colors = get_block_colors(atomic_num)
fig = plot_reduced_dimension(dc.reduced_dimensions, color = block_colors, suptitle="By metal center")
fig.savefig(f"{out_dir}/metal_center_dblock.png", dpi = 300)



tm_cmap, norm = get_tm_colormap()



fig = plot_reduced_dimension(dc.reduced_dimensions, color =atomic_num, suptitle="By metal center", cmap = tm_cmap, norm = norm)

fig.savefig(f"{out_dir}/metal_center_dblock.png", dpi = 300)




cns = get_coordination_numbers(dataset.molecules)

fig = plot_reduced_dimension(dc.reduced_dimensions, color =cns)
fig.savefig(f"{out_dir}/coordination_number.png", dpi = 300)


fig = plot_reduced_dimension_3d(dc.reduced_dimensions, color =atomic_num, suptitle="By metal center", cmap = tm_cmap, norm = norm)
fig.savefig(f"{out_dir}/metal_center_dblock3d.png", dpi = 300)

