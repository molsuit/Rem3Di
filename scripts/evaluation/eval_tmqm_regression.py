
from threedscriptors.evaluation.clustering import PCACalculator, UMAPCalculator
from threedscriptors.evaluation.evaluation_pipeline import (
    RegressionTestTask,
    EvalPipelineRunner,
)
from threedscriptors.configuration.data_config import DatasetSplit
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.data_handling.pipelines import reload_dataset_pipeline

from threedscriptors.evaluation.clustering.tmqm_clustering_utils import get_coordination_numbers, get_metal_center_type, get_tm_colormap, get_atomic_num_colors, get_block_colors

model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/149-2025_07_31_12_53_12-TMQMRegressionTraining"


#model_directory = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/144-2025_07_30_14_47_16-TMQM First Run"

model = ModelBuilder.from_directory(model_directory).build_model()



train_dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/tmqm_training"
)

out_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/eval_runs/tmqm_training"

#dataset = reload_dataset_pipeline(train_dataset_directory).build()
#
#reg = RegressionTestTask(dataset)
#runner = EvalPipelineRunner(
#    tasks=[reg], dataset_name="tmqm", dataset_split=DatasetSplit.TRAIN
#)
#
#
#runner.evaluate(model)
#runner.output_results(
#    output_directory=f"{out_dir}/train",
#    model_name="tmqm_regressionmodel",
#)



test_dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/tmqm_test"
)



dataset = reload_dataset_pipeline(test_dataset_directory).build()
reg = RegressionTestTask(dataset)
runner = EvalPipelineRunner(
    tasks=[reg], dataset_name="tmqm", dataset_split=DatasetSplit.TEST
)

runner.evaluate(model)
runner.output_results(
    output_directory=f"{out_dir}/test",
    model_name="tmqm_regressionmodel",
)