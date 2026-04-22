from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset_creation.generators.tsv_generator import (
    TSVMoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    ConformerGenerationStage,
    CopyDataStage,
)

tsv_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/BindingDB_All.tsv"

creation_config = DatasetCreationConfig(
    path=Path(
        "/share/snw30/projects/threedscriptor/fast_data_preparation/data/binding_db"
    ),
    N_structures=1000,
    max_embed_attempts=100,
    max_MMFF_steps=100,
)
dataset_config = DatasetConfig()

mol_generator = TSVMoleculeGenerator(tsv_file=tsv_path, batch_size=4)

conformal_stage = ConformerGenerationStage(dataset_creation_config=creation_config)
copy_data = CopyDataStage(dtype=torch.float32)

pipeline = [conformal_stage, copy_data]


orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=mol_generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
