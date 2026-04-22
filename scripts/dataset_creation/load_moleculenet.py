from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset.tasks import (
    TaskSet,
)
from threedscriptors.data_handling.dataset_creation.generators.moleculenet_generator import (
    MoleculeNetGenerator,
    MoleculeNetTask,
    convert_tasks_to_configs,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    ConformerGenerationStage,
    CopyDataStage,
)

mnet_dir = Path(
    "/share/snw30/projects/threedscriptor/raw_datasets/molecule_net/processed"
)

creation_config = DatasetCreationConfig(
    path=Path(
        "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net"
    ),
    N_structures=100_000,
    max_embed_attempts=10_000,
    max_MMFF_steps=100,
)

molecule_net_tasks = [
    MoleculeNetTask.BACE,
    MoleculeNetTask.BBBP,
    MoleculeNetTask.HIV,
    MoleculeNetTask.ESOL,
    MoleculeNetTask.FREE_SOLVE,
    MoleculeNetTask.LIPOPHILICITY,
]


task_configs = convert_tasks_to_configs(molecule_net_tasks)

gen = MoleculeNetGenerator(mnet_dir, batch_size=500, tasks=task_configs, max_atoms=100)

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
