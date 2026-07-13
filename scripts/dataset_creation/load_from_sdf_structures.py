from pathlib import Path

import torch

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from remedi.data_handling.dataset.tasks import ElementSet
from remedi.data_handling.dataset_creation.generators.sdf_generator import (
    SDFMoleculeGenerator,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

sdf_file = Path("/scratch/s5f/wedigs.s5f/raw_datasets/pcqm4m/pcqm4m-v2-train.sdf")

mol_generator = SDFMoleculeGenerator(sdf_file=sdf_file, loading_batch_size=10000)

filter_stage = FilterAtomsStage(
    config=FilterAtomsStageConfig(element_set=ElementSet.mace_off)
)
copy_data = CopyDataStage(dtype=torch.float64)

creation_config = DatasetCreationConfig(
    path=Path("/scratch/s5f/wedigs.s5f/datasets/pcqm4m/pcqm_only_structures_3_5_M"),
    N_structures=3_500_000,
)
# Defaults shard correctly (small read chunks, few on-disk shard files);
# the old atom_chunk=450/molecule_chunk=50 were a one-file-per-chunk
# workaround that no longer applies under zarr v3 sharding.
dataset_config = DatasetConfig()

orchestrator = DatasetConstructionOrchestrator(
    pipeline=[filter_stage, copy_data],
    batch_generator=mol_generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
