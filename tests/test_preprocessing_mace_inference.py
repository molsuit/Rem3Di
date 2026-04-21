"""Cross-check MaceModel inference in the preprocessor against the ASE calculator.

The `PreprocessorWithAtomicEmbedding` path builds a torch-sim `SimState`
directly from a batched `Sample` and reads `node_feats` out of the model. The
ASE `MACECalculator.get_descriptors(..., invariants_only=False)` returns the
same `node_feats` (concatenation of per-layer products). Running both over the
same geometry must yield the same descriptors up to floating-point tolerance.
"""

from __future__ import annotations

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
