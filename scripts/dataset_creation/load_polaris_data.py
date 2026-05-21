from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset.tasks import (
    TaskSet,
)
from threedscriptors.data_handling.dataset_creation.generators.polaris_generator import (
    PolarisGenerator,
    get_polaris_task_configs,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    ConformerGenerationStage,
    CopyDataStage,
)

dataset_name = "antiviral_potency"


creation_config = DatasetCreationConfig(
    path=Path(
        f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/{dataset_name}"
    ),
    N_structures=10000,
)


task_configs = get_polaris_task_configs(dataset_name)

gen = PolarisGenerator(dataset_name=dataset_name, batch_size=500, max_atoms=100)

conformal_stage = ConformerGenerationStage(dataset_creation_config=creation_config)
copy_data = CopyDataStage(dtype=torch.float64)

pipeline = [conformal_stage, copy_data]

dataset_config = DatasetConfig(
    atom_chunk=450,
    molecule_chunk=50,
    contains_smiles=True,
    tasks=TaskSet.from_list(task_configs),
)

print(dataset_config)


orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=gen,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
