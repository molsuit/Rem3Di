from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.pipelines import (
    regression_training_from_structures_pipeline,
)
from threedscriptors.data_handling.source_preprocessing.tmqm_preprocessing import (
    load_tmqm_dataset,
)

directory = (
    "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/tmqm/"
)

tasks = ["HL_Gap"]
csd_ids_int, molecules, regression_targets, regression_masks, tasks = load_tmqm_dataset(
    f"{directory}/raw_data/", tasks
)


MACE_PATH = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)

embedding_model_config = MaceCalculatorConfig(
    mace_calc=None,
    model_name="medium",
    model_path=MACE_PATH,
    enable_cueq=True,
    device="cuda",
)


dataset_config = DatasetConfig(
    N_molecules=10,
    dataset_type=DatasetTypes.REGRESSION,
    BFGS_tol=0.2,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
    tasks=tasks,
)


dataset = regression_training_from_structures_pipeline(
    dataset_config=dataset_config,
    molecules=molecules,
    mol_ids=csd_ids_int,
    regression_targets=regression_targets,
    regression_masks=regression_masks,
).build()


print(dataset.embeddings.dtype)

# store_data_to_disk(dataset, directory)
