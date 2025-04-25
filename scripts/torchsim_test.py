import time

import torch
import torch_sim as ts
from ase import Atoms
from ase.optimize import LBFGS
from mace.calculators import MACECalculator, mace_mp
from torch_sim.models import MaceModel

from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
)
from threedscriptors.data_handling.data_build_pipeline import (
    ConformalEmbeddingStage,
    InitializeBuildPipeline,
    InsertSmilesStage,
    PipelineOrchestrator,
)
from threedscriptors.data_handling.dataset import RegressionDataset
from threedscriptors.data_handling.source_preprocessing.similarity_screening_preprocessing import (
    load_similarity_screening_data,
)


def relax_structures_mace_calc(mace_calculator, molecules: list[Atoms]):
    new_mols = []
    for mol in molecules:
        mol.calc = mace_calculator
        opt = LBFGS(mol)
        opt.run(fmax=0.5, steps=300)

        new_mols.append(mol)

    return new_mols


def relax_structures_torchsim(mace_model: MaceModel, molecules: list[Atoms]):
    state = ts.initialize_state(molecules, "cuda", torch.float64)

    print(state.device)
    print(mace_model.device)

    fire_init, fire_update = ts.optimizers.fire(mace_model)
    fire_state = fire_init(state)

    # Initialize the batcher
    batcher = ts.InFlightAutoBatcher(
        model=mace_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=None,
        max_iterations=100,  # Optional: maximum convergence attempts per state
    )
    # Load states
    batcher.load_states(fire_state)

    convergence_fn = ts.generate_force_convergence_fn(1e-1)

    # Process states until all are complete
    all_converged_states, convergence_tensor = [], None
    while (result := batcher.next_batch(fire_state, convergence_tensor))[0] is not None:
        # collect the converged states
        fire_state, converged_states = result
        all_converged_states.extend(converged_states)

        # optimize the batch, we stagger the steps to avoid state processing overhead
        for _ in range(10):
            fire_state = fire_update(fire_state)

        # Check which states have converged
        convergence_tensor = convergence_fn(fire_state, None)
        print(f"Convergence tensor: {batcher.current_idx}")

    else:
        all_converged_states.extend(result[1])

    # Restore original order
    final_states = batcher.restore_original_order(all_converged_states)

    return final_states


directory = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/similarity_screening_datasets"

N_classes = 1

smiles, class_labels, activity_labels, target_class_dict = (
    load_similarity_screening_data(dir_path=directory, N_target_classes=N_classes)
)

MACE_PATH = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)
embedding_model_config = MaceCalculatorConfig(
    mace_calc=MACECalculator(model_paths=MACE_PATH, enable_cueq=False, device="cuda"),
    model_name="mace_mp_medium",
    model_path=MACE_PATH,
    enable_cueq=False,
    device="cpu",
)

dataset_config = DatasetConfig(
    N_molecules=None,
    dataset_type=DatasetTypes.REGRESSION,
    BFGS_tol=0.05,
    BFGS_max_steps=500,
    N_conformers=1,
    embedding_model_config=embedding_model_config,
    max_atoms=None,
)


stages = [
    InitializeBuildPipeline(dataset_config, RegressionDataset),
    InsertSmilesStage(smiles=smiles),
    ConformalEmbeddingStage(),
]

dataset = PipelineOrchestrator(stages).build()

molecules = dataset.molecules[:100]

molecules1 = molecules.copy()

print(molecules[0].get_cell())
mace = mace_mp("medium", return_raw_model=True, device="cuda")

torch_sim_mace_model = MaceModel(mace, enable_cueq=False, device="cuda")

ts_start_time = time.time()
relaxed_molecules = relax_structures_torchsim(torch_sim_mace_model, molecules)

ts_end_time = time.time()

print(ts_end_time - ts_start_time)


# mace_calc = MACECalculator(model_paths=MACE_PATH, enable_cueq= False, device="cuda")
#
# ase_start_time = time.time()
# relaxed_molecules1 = relax_structures_mace_calc(mace_calc, molecules1)
#
# ase_end_time = time.time()
#
# print(ase_end_time -ase_start_time)
