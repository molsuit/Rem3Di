"""Cross-check MaceModel inference in the preprocessor against the ASE calculator.

The `PreprocessorWithAtomicEmbedding` path builds a torch-sim `SimState`
directly from a batched `Sample` and reads `node_feats` out of the model. The
ASE `MACECalculator.get_descriptors(..., invariants_only=False)` returns the
same `node_feats` (concatenation of per-layer products). Running both over the
same geometry must yield the same descriptors up to floating-point tolerance.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule

from threedscriptors.configuration.mace_config import MaceConfig
from threedscriptors.data_handling.sample import Sample


def _sample_from_atoms(atoms: Atoms, device: torch.device) -> Sample:
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float64, device=device)
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(), dtype=torch.long, device=device
    )
    system_index = torch.zeros(len(atoms), dtype=torch.long, device=device)
    total_charge = torch.zeros(1, dtype=torch.float64, device=device)
    total_spin = torch.ones(1, dtype=torch.float64, device=device)
    return Sample(
        atomic_positions=positions,
        atomic_numbers=atomic_numbers,
        system_index=system_index,
        total_charge=total_charge,
        total_spin=total_spin,
    )


@pytest.mark.parametrize("mol_name", ["H2O", "CH4", "NH3"])
def test_torchsim_matches_ase_descriptors(mol_name: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MaceConfig(
        device=str(device),
        dtype="float64",
        compute_forces=False,
        compute_stress=False,
        enable_cueq=False,
    )

    from threedscriptors.model.preprocessing.preprocessing import _sample_to_simstate

    atoms = molecule(mol_name)
    sample = _sample_from_atoms(atoms, device=device)

    torch_sim_model = config.build_torch_sim_model()
    state, _ = _sample_to_simstate(sample, r_max=float(torch_sim_model.r_max))
    with torch.inference_mode():
        out = torch_sim_model(state)
    ts_descriptors = out["node_feats"].detach().cpu().numpy()

    ase_calculator = config.build_ase_calculator()
    ase_descriptors = ase_calculator.get_descriptors(atoms, invariants_only=False)

    assert ts_descriptors.shape == ase_descriptors.shape
    np.testing.assert_allclose(ts_descriptors, ase_descriptors, atol=1e-6, rtol=1e-5)


_POLAR_MODEL_PATH = Path(
    os.environ.get(
        "MACE_POLAR_MODEL_PATH",
        "/home/steffen/projects/mol_descriptors/mace_model/MACE-POLAR-1-M.model",
    )
)


@pytest.mark.skipif(
    os.environ.get("RUN_POLAR_TESTS") != "1",
    reason="PolarMACE is heavy (SCF + k-space on full-size model). "
    "Set RUN_POLAR_TESTS=1 on a machine with enough RAM/VRAM to opt in.",
)
@pytest.mark.skipif(
    not _POLAR_MODEL_PATH.exists(),
    reason=f"POLAR checkpoint not available at {_POLAR_MODEL_PATH} "
    "(set MACE_POLAR_MODEL_PATH to override)",
)
@pytest.mark.parametrize("mol_name", ["H2O", "CH4", "NH3"])
def test_polar_run_mace_matches_ase(mol_name: str) -> None:
    """PolarMACE can't go through torch_sim.MaceModel.forward (it requires
    fermi_level / external_field / rcell / volume, which the wrapper doesn't
    propagate). PreprocessorWithAtomicEmbedding._run_mace builds the data_dict
    itself and should match the ASE calculator's node_feats bit-for-bit within
    float32 tolerance.

    The ASE calc is built once to derive the reference data_dict and then
    released before instantiating the torch-sim model, so the PolarMACE weights
    are only resident once at a time (this test is still RAM-hungry).
    """
    from threedscriptors.configuration.architecture_config import (
        EmbeddingPreprocessConfig,
    )
    from threedscriptors.model.preprocessing.geometric_preprocessor import (
        PairDistanceMatrixGeometricPreprocessor,
    )
    from threedscriptors.model.preprocessing.preprocessing import (
        PreprocessorWithAtomicEmbedding,
        _sample_to_simstate,
    )
    from threedscriptors.model.preprocessing.radial_basis_functions import (
        BesselBasisFunctions,
    )
    from threedscriptors.utils.model_utils import get_mace_model_irrep_signature

    # POLAR's SCF + k-space machinery is heavy; pin to CPU for determinism.
    device = torch.device("cpu")
    config = MaceConfig(
        model_path=_POLAR_MODEL_PATH,
        device=str(device),
        dtype="float32",
        compute_forces=False,
        compute_stress=False,
        enable_cueq=False,
    )

    # Non-periodic systems: hand MACE a unit cell (ASE's default for pbc=False).
    # The big extent-sized cell is only needed as a bounding box for
    # torch-sim's neighbor list; with pbc=False, cell magnitude doesn't change
    # the physics, but a big cell explodes PolarMACE's k-space grid.
    atoms = molecule(mol_name)
    ase_calc = config.build_ase_calculator()
    r_max = float(ase_calc.models[0].r_max)
    atoms.set_cell(np.eye(3))
    atoms.pbc = False
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.info["external_field"] = [0.0, 0.0, 0.0]
    atoms.info["fermi_level"] = 0.0

    batch = ase_calc._atoms_to_batch(atoms)
    ase_out = ase_calc.models[0](
        batch.to_dict(), compute_force=False, compute_stress=False
    )
    ase_feats = ase_out["node_feats"].detach().cpu().numpy()

    # Drop the ASE calc and its model copy before building the torch-sim side.
    import gc

    del batch, ase_out, ase_calc
    gc.collect()

    torch_sim_model = config.build_torch_sim_model()
    sample_positions = torch.tensor(
        atoms.get_positions(), dtype=torch.float32, device=device
    )
    sample = Sample(
        atomic_positions=sample_positions,
        atomic_numbers=torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.int32, device=device
        ),
        system_index=torch.zeros(len(atoms), dtype=torch.long, device=device),
        total_charge=torch.zeros(1, dtype=torch.float32, device=device),
        total_spin=torch.ones(1, dtype=torch.float32, device=device),
    )
    input_irreps = str(get_mace_model_irrep_signature(torch_sim_model.model))
    emb_cfg = EmbeddingPreprocessConfig(
        input_irreps=input_irreps,
        pseudoscalar_dimension=32,
        chiral_embedding_dimension=32,
    )
    radial = BesselBasisFunctions(8, 6.0)
    geo = PairDistanceMatrixGeometricPreprocessor(
        radial_basis=radial,
        N_radial_basis_functions=8,
        distance_cutoff=6.0,
        d_projection=32,
    )
    preprocessor = PreprocessorWithAtomicEmbedding(
        mace_model=torch_sim_model,
        atomic_preprocessor=emb_cfg.build(),
        geometric_preprocessor=geo,
    ).to(device)

    state, _ = _sample_to_simstate(sample, r_max=r_max)
    with torch.inference_mode():
        our_feats = preprocessor._run_mace(state).detach().cpu().numpy()

    assert our_feats.shape == ase_feats.shape
    np.testing.assert_allclose(our_feats, ase_feats, atol=1e-5, rtol=1e-4)
