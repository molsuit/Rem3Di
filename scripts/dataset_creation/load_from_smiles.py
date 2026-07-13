from pathlib import Path

import torch

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterMoleculeStageConfig,
)
from remedi.data_handling.dataset_creation.generators.smiles_list_generator import (
    SmilesMoleculeGenerator,
    open_smiles_file,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    ConformerGenerationStage,
    CopyDataStage,
    FilterMoleculeStage,
)

smiles_file = Path(
    "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/scripts/dataset_creation/pharma_smiles"
)


smiles = open_smiles_file(smiles_file)

creation_config = DatasetCreationConfig(
    path=Path(
        "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/pharma"
    ),
    N_structures=len(smiles),
)


gen = SmilesMoleculeGenerator(smiles, batch_size=500)

filter_stage = FilterMoleculeStage(config=FilterMoleculeStageConfig(max_atoms=100))
conformal_stage = ConformerGenerationStage(dataset_creation_config=creation_config)
copy_data = CopyDataStage(dtype=torch.float64)

pipeline = [filter_stage, conformal_stage, copy_data]

dataset_config = DatasetConfig(
    atom_chunk=450,
    molecule_chunk=50,
    contains_smiles=True,
    tasks=None,
)

print(dataset_config)


orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=gen,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
