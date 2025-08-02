from mace.calculators import MACECalculator

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.dataset import RegressionDatasetwithPositions
from threedscriptors.data_handling.dataset_builder import DatasetBuilder
from threedscriptors.data_handling.dataset_io import store_data_to_disk
from threedscriptors.data_handling.pipelines import (
    regression_training_from_structures_pipeline,
)
from threedscriptors.data_handling.source_preprocessing.tmqm_preprocessing import (
    TmqmTask,
    load_tmqm_dataset,
)
from threedscriptors.training.dataset_splitting import DatasetSplitting

dataset_directory = (
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/tmqm/"
)

tasks = [TmqmTask.HL_GAP]


structure_ids, molecules, regression_targets, regression_masks, tasks = load_tmqm_dataset(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/tmqm", tasks)


MACE_PATH = (
    "/share/snw30/projects/mace_model/mace_agnesi_medium.model"
)



embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda"),
    model_name="mace_mp",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)

dataset_config = DatasetConfig(
    N_molecules=structure_ids[-1].molecule_id,
    dataset_type=RegressionDatasetwithPositions,
    BFGS_tol=0.1,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
    only_heavy_atoms=False,
    dataset_name="tmqm",
)


dataset = regression_training_from_structures_pipeline(
    dataset_config=dataset_config,
    molecules=molecules,
    structure_ids=structure_ids,
    regression_targets=regression_targets,
    regression_masks=regression_masks,
).build()

store_data_to_disk(dataset, dataset_directory)





ds = DatasetSplitting(dataset)
names = ["training", "test"]
split_ratios = [0.9, 0.1]
split_dataset_indices = ds.general_split(split_ratios, True)


for ids, name in zip(split_dataset_indices,names, strict=False):
    new_dataset = ds.materialise_dataset_split(dataset, ids)

    db = DatasetBuilder(new_dataset)
    db.canonicalize_structure_ids()
    store_data_to_disk(new_dataset, dataset_directory+"_" + name)
