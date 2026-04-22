from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset_creation.generators.geom_generator import (
    GeomGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
)

geom_dir = Path("/share/snw30/projects/threedscriptor/raw_datasets/geom_drugs")

mol_generator = GeomGenerator(
    geom_dir=geom_dir,
    boltzman_weight_threshold=0.05,
    max_atoms=128,
    loading_batch_size=500,
)

pipeline = [CopyDataStage(dtype=torch.float64)]

creation_config = DatasetCreationConfig(
    path=Path(
        "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/geom_drugs"
    ),
    N_structures=3_000_000,
)
dataset_config = DatasetConfig(atom_chunk=450, molecule_chunk=50)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=mol_generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
