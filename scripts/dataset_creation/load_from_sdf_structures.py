from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset_creation.generators.sdf_generator import (
    SDFMoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
)

sdf_files = "/scratch/public/snw30/dataset/pcqm/pcqm4m-v2-train.sdf"

mol_generator = SDFMoleculeGenerator(sdf_file=sdf_files, loading_batch_size=10000)

copy_data = CopyDataStage(dtype=torch.float64)

creation_config = DatasetCreationConfig(
    path=Path("/scratch/public/snw30/dataset/pcqm/pcqm_only_structures_3_5_M"),
    N_structures=3_500_000,
)
dataset_config = DatasetConfig(
    atom_chunk=450,
    molecule_chunk=50,
)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=[copy_data],
    batch_generator=mol_generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
