from pathlib import Path

import torch
from mace.calculators.foundations_models import mace_mp
from torch_sim.models.mace import MaceModel

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
    BatchedEmbeddingStage,
)
from threedscriptors.utils.model_utils import get_mace_model_irrep_signature

xyz_files = [
    str(f)
    for f in Path("/share/snw30/projects/threedscriptor/raw_datasets/tmqm").glob(
        "*.xyz"
    )
]



generator = XYZMoleculeGenerator(xyz_file=xyz_files, loading_batch_size=100)

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the MACE "small" foundation model
mace = mace_mp(model = "medium", default_dtype= "float64", device = "cuda", enable_cueq = True, return_raw_model=True)
mace_irreps = get_mace_model_irrep_signature(mace)
#egret_irreps = Irreps("192x0e+192x1o+192x2e+192x0e")


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

# If you want to change what is loaded from MACE, you have to write a PipelineStage for your use case. You might have to modify torch_sim.models.mace.MaceModel.forward to yield the atomic energies.

pipeline = [batched_embedding]

creation_config = DatasetCreationConfig(
    path=Path("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/tmqm"),
    N_structures=5_000,
)
dataset_config = DatasetConfig(embedding_dim=mace_irreps.dim ,irreps = mace_irreps, atom_chunk=450, molecule_chunk=50, contains_smiles= False)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=[batched_embedding],
    batch_generator=generator,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
