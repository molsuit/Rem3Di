
from pathlib import Path

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.descriptor_analysis.clustering import UMAPCalculator
from threedscriptors.evaluation.descriptor_analysis.clustering_task import (
    DescriptorClusteringTask,DescriptorElementAnalysis
)
from threedscriptors.evaluation.evaluation_pipeline import EvalPipelineRunner
from threedscriptors.model.model_builder import ModelBuilder

dataset_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/geom_drugs"
)
model_dir = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/geom_drugs_350k/1-2025_10_12_18_02_05-Train"
)
model_name = "GEOM_DRUGS"
eval_dir = Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/geom_drugs_pretraining/geom_drugs")

remedi_model = ModelBuilder.from_directory(model_dir).build_remedi_model()



dataset = MoleculeDataset.open_existing_dataset_from_dir(dataset_dir)

clustering_calculator = UMAPCalculator()
clustering_task = DescriptorClusteringTask(dataset=dataset, clustering_calculator=clustering_calculator)

capacity_diagnostic_task= DescriptorElementAnalysis(dataset)

eval_pipeline = EvalPipelineRunner(tasks = [clustering_task, capacity_diagnostic_task], dataset_name= "pcqm_benchmark")

eval_pipeline.evaluate(remedi_model, model_name)
eval_pipeline.output_results(output_directory=eval_dir)
