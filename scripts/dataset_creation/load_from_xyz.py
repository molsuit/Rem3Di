from pathlib import Path

import torch

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
)
from remedi.data_handling.dataset_creation.generators.xyz_generator import (
    XYZMoleculeGenerator,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

xyz_files = [
    Path(f)
    for f in Path("/path/to/raw_datasets/omol25_4M_train_tmcs").glob(
        "*.extxyz"
    )
]


generator = XYZMoleculeGenerator(
    xyz_file=xyz_files,
    loading_batch_size=10000,
    charge_key="charge",
    spin_key="spin",
)

filter_stage = FilterAtomsStage(
    config=FilterAtomsStageConfig(
        max_atoms=200,
        reject_zero_h=True,
        min_h_heavy_ratio=0.1,
    )
)
pipeline = [filter_stage, CopyDataStage(dtype=torch.float64)]

creation_config = DatasetCreationConfig(
    path=Path("/path/to/datasets/omol25_tmcs"),
    N_structures=None,
)
# Defaults shard correctly (small read chunks, few on-disk shard files);
# the old atom_chunk=450/molecule_chunk=50 were a one-file-per-chunk
# workaround that no longer applies under zarr v3 sharding.
dataset_config = DatasetConfig(contains_smiles=False)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
