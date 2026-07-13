"""Build the GEOM-Drugs zarr on JUWELS. Routes SMILES through the same
standardization (strip_salts / neutralize / mace_off element filter / dedupe)
that the benchmark loaders use, so pretrain canonical SMILES match the eval
canonical SMILES for DatasetComparison."""

from pathlib import Path

import torch

from remedi.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
    FilterMoleculeStageConfig,
)
from remedi.data_handling.dataset.tasks import ElementSet
from remedi.data_handling.dataset_creation.generators.geom_generator import (
    GeomGenerator,
)
from remedi.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from remedi.data_handling.dataset_creation.pipeline_stages import (
    CopyDataStage,
)

geom_dir = Path("/path/to/raw_datasets/geom_drugs")

smiles_filter = FilterMoleculeStageConfig(
    max_atoms=100,
    element_set=ElementSet.mace_off,
    strip_salts=True,
    neutralize=True,
    dedupe=True,
)

mol_generator = GeomGenerator(
    geom_dir=geom_dir,
    boltzman_weight_threshold=0.05,
    max_atoms=128,
    loading_batch_size=500,
    filter_config=smiles_filter,
)

pipeline = [CopyDataStage(dtype=torch.float64)]

creation_config = DatasetCreationConfig(
    path=Path("/path/to/datasets/geom_drugs"),
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
