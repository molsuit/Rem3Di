from pathlib import Path

import torch
from mace.calculators.foundations_models import mace_off
from torch_sim.models.mace import MaceModel

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
    BatchedEmbeddingStage,
    CopyDataStage,
)
from threedscriptors.utils.model_utils import get_mace_model_irrep_signature

sdf_files = "/scratch/public/snw30/dataset/pcqm/pcqm4m-v2-train.sdf"

mol_generator = SDFMoleculeGenerator(sdf_file=sdf_files, loading_batch_size=10000)


# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the MACE "small" foundation model
mace = mace_off(
    model="/share/snw30/projects/mace_model/MACE-OFF24_medium.model",
    default_dtype="float64",
    device=device,
    enable_cueq=True,
    return_raw_model=True,
)

mace_irreps = get_mace_model_irrep_signature(mace)
# egret_irreps = Irreps("192x0e+192x1o+192x2e+192x0e")


mace_model = MaceModel(
    model=mace,
    device=device,
    dtype=torch.float64,
    compute_forces=False,
    compute_stress=False,
    compute_descriptors=True,
    enable_cueq=True,
)

batched_embedding = BatchedEmbeddingStage(
    mace_model, device=device, dtype=torch.float64
)
copy_data = CopyDataStage(dtype=torch.float64)

# pipeline = [batched_embedding]

creation_config = DatasetCreationConfig(
    path=Path("/scratch/public/snw30/dataset/pcqm/pcqm_only_structures_3_5_M"),
    N_structures=3_500_000,
)
dataset_config = DatasetConfig(
    embedding_dim=mace_irreps.dim,
    irreps=mace_irreps,
    atom_chunk=450,
    molecule_chunk=50,
    contains_embeddings=False,
)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=[copy_data],
    batch_generator=mol_generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
