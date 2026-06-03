"""Build a 100k-structure pcqm4m subset on JUWELS for fast DatasetComparison
iteration. Mirrors load_from_sdf_structures.py but with /p/scratch paths and
N_structures=100_000. The SDFMoleculeGenerator stops after the orchestrator
has accepted N_structures rows, so this only ever reads the first ~100k
molecules of the SDF -- much faster than the full 3.5M build.

Run with:
    uv run python scripts/dataset_creation/load_pcqm100k_juwels.py
"""

from pathlib import Path

import torch

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterAtomsStageConfig,
    FilterMoleculeStageConfig,
)
from threedscriptors.data_handling.dataset.tasks import ElementSet
from threedscriptors.data_handling.dataset_creation.generators.sdf_generator import (
    SDFMoleculeGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
    FilterAtomsStage,
)

sdf_file = Path(
    "/p/scratch/mace/wedig1/raw_datasets/pcqm4m/extracted/pcqm4m-v2-train.sdf"
)

# Match the benchmark filter config (configs/dataset_creation/benchmarks_local.yaml)
# so pretrain SMILES go through the same canonicalization as eval SMILES.
smiles_filter = FilterMoleculeStageConfig(
    max_atoms=100,
    element_set=ElementSet.mace_off,
    strip_salts=True,
    neutralize=True,
    dedupe=True,
)

mol_generator = SDFMoleculeGenerator(
    sdf_file=sdf_file,
    loading_batch_size=10000,
    filter_config=smiles_filter,
)

filter_stage = FilterAtomsStage(
    config=FilterAtomsStageConfig(element_set=ElementSet.mace_off)
)
copy_data = CopyDataStage(dtype=torch.float64)

creation_config = DatasetCreationConfig(
    path=Path("/p/scratch/mace/wedig1/datasets/pcqm4m/pcqm100k_std"),
    N_structures=100_000,
)
dataset_config = DatasetConfig()

orchestrator = DatasetConstructionOrchestrator(
    pipeline=[filter_stage, copy_data],
    batch_generator=mol_generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)

orchestrator.build_dataset()
