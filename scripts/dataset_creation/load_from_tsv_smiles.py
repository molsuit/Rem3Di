from pathlib import Path

import torch
from dataset_prep.configuration import DatasetConfig, DatasetCreationConfig
from dataset_prep.dataset_creation.molecule_generators import (
    TSVMoleculeGenerator,
)
from dataset_prep.dataset_creation.orchestrator import DatasetConstructionOrchestrator
from dataset_prep.dataset_creation.pipeline_stages import (
    BatchedEmbeddingStage,
    ConformerGenerationStage,
    ParallelRelaxStage,
)
from mace.calculators import mace_off
from torch_sim.models.mace import MaceModel

tsv_path = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/BindingDB_All.tsv"

creation_config = DatasetCreationConfig(
    path=Path(
        "/share/snw30/projects/threedscriptor/fast_data_preparation/data/binding_db"
    ),
    N_structures=1000,
    max_embed_attempts=100,
    max_MMFF_steps=100,
)
dataset_config = DatasetConfig(embedding_dim=640)

mol_generator = TSVMoleculeGenerator(tsv_file=tsv_path, batch_size=4)


# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

mace = mace_off(
    model="/share/snw30/projects/mace_model/MACE-OFF24_medium.model",
    default_dtype="float32",
    device="cuda",
    enable_cueq=True,
    return_raw_model=True,
)

# Note that we use two different torch-sim mace models here, one that computes forces and one that doesnt. Disbaling force computation reduces the inference cost because we drop a backward pass
mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float32,
    compute_forces=False,
    compute_stress=False,
    compute_descriptors=True,
    enable_cueq=False,
)

conformal_stage = ConformerGenerationStage(dataset_creation_config=creation_config)
batched_embedding = BatchedEmbeddingStage(
    mace_model, device=device, dtype=torch.float32
)

mace_model_with_force = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float32,
    compute_forces=True,
    compute_stress=False,
    enable_cueq=False,
)

# This is a fixed N_step relaxation (Should be changed to have a AutoBatching/HotSwapping)
relax_stage = ParallelRelaxStage(
    mace_model_with_force, device, dtype=torch.float32, N_steps=50
)

pipeline = [conformal_stage, relax_stage, batched_embedding]


orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=mol_generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
