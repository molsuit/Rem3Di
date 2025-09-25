from pathlib import Path

import torch
from mace.calculators.foundations_models import mace_off
from torch_sim.models.mace import MaceModel

from threedscriptors.configuration.dataset_config import (
    DatasetConfig,
    DatasetCreationConfig,
)
from threedscriptors.data_handling.dataset.tasks import (
    TaskConfig,
    TaskScope,
    TaskSet,
    TaskType,
)
from threedscriptors.data_handling.dataset_creation.generators.moleculenet_generator import (
    MoleculeNetGenerator,
)
from threedscriptors.data_handling.dataset_creation.orchestrator import (
    DatasetConstructionOrchestrator,
)
from threedscriptors.data_handling.dataset_creation.pipeline_stages import (
    BatchedEmbeddingStage,
    ConformerGenerationStage,
)
from threedscriptors.utils.model_utils import get_mace_model_irrep_signature

file = "/share/snw30/projects/threedscriptor/raw_datasets/molecule_net/processed/lipophilicity.csv"


creation_config = DatasetCreationConfig(
    path=Path(
        "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/datasets/molecule_net"
    ),
    N_structures=4200,
    max_embed_attempts=100,
    max_MMFF_steps=100,
)


gen = MoleculeNetGenerator(file, 8, mol_column="smiles", tasks=["lipophilicity"], max_atoms=100)


# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the MACE "small" foundation model
mace = mace_off(
    model="/share/snw30/projects/mace_model/MACE-OFF24_medium.model",
    default_dtype="float64",
    device="cuda",
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
conformal_stage = ConformerGenerationStage(dataset_creation_config=creation_config)

# If you want to change what is loaded from MACE, you have to write a PipelineStage for your use case. You might have to modify torch_sim.models.mace.MaceModel.forward to yield the atomic energies.

pipeline = [conformal_stage, batched_embedding]

tasks = [TaskConfig(name= "lipophilicity",task_type=TaskType.regression, scope=TaskScope.system)]
dataset_config = DatasetConfig(
    embedding_dim=mace_irreps.dim,
    irreps=mace_irreps,
    atom_chunk=450,
    molecule_chunk=50,
    contains_smiles=True,
    tasks = TaskSet.from_list(task_list=tasks)
)

orchestrator = DatasetConstructionOrchestrator(
    pipeline=pipeline,
    batch_generator=gen,
    construction_config=creation_config,
    dataset_config=dataset_config,
)


orchestrator.build_dataset()
