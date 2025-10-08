
from pathlib import Path

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.data_handling.dataset.training_dataset import (
    TrainingMoleculeDataset,
    pos_emb_getitem,
)
from threedscriptors.evaluation.descriptor_analysis.clustering import UMAPCalculator
from threedscriptors.evaluation.descriptor_analysis.clustering_task import (
    DescriptorClusteringTask,
)
from threedscriptors.evaluation.evaluation_pipeline import EvalPipelineRunner
from threedscriptors.model.model_builder import ModelBuilder

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/geom_drugs"
)
model_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/small_geom_drugs/4-2025_10_06_22_47_28-Train"
)

eval_dir = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/geom_drugs_pretraining")

remedi_model = ModelBuilder.from_directory(model_dir).build_remedi_model()


dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)

ds = TrainingMoleculeDataset(dataset_dir, get_item=pos_emb_getitem)

clustering_calculator = UMAPCalculator()

task = DescriptorClusteringTask(dataset=ds, clustering_calculator=clustering_calculator)

eval_pipeline = EvalPipelineRunner(tasks = [task], dataset_name= "GeomDrugs")

eval_pipeline.evaluate(remedi_model)
eval_pipeline.output_results(output_directory=eval_dir, model_name="pretrained_geom_drugs")