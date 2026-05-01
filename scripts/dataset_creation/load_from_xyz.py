from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset_creation.generators.xyz_generator import (
    XYZMoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
)

xyz_files = [
    Path(f)
    for f in Path("/scratch/s5f/wedigs.s5f/raw_datasets/tmqm").glob("*.xyz")
]


generator = XYZMoleculeGenerator(
    xyz_file=xyz_files,
    loading_batch_size=100,
    charge_key="q",
    spin_key="S",
    max_atoms=200,
    reject_zero_h=True,
    min_h_heavy_ratio=0.1,
)

pipeline = [CopyDataStage(dtype=torch.float64)]

creation_config = DatasetCreationConfig(
    path=Path("/scratch/s5f/wedigs.s5f/datasets/tmqm"),
    N_structures= None,
)
dataset_config = DatasetConfig(
    atom_chunk=450,
    molecule_chunk=50,
    contains_smiles=False,
)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
