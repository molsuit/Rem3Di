"""MACE node-feature handling in the on-the-fly preprocessor: cuequivariance
layout, per-system padding and centring."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ase.build import molecule
from e3nn import o3
from e3nn.o3 import Irreps

from remedi.configuration.mace_config import MaceConfig
from remedi.model.preprocessing.preprocessing import (
    _capture_node_feats,
    _sample_to_simstate,
    centre_per_system,
    ir_mul_to_mul_ir,
    pad_per_system,
)
from remedi.training.data.sample import Sample, yield_molecules_collate_fn

MACE_OFF_MEDIUM = Path.home() / ".cache" / "mace" / "MACE-OFF23_medium.model"


def _mul_ir_to_ir_mul(features: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    blocks = []
    for (multiplicity, irrep), block_slice in zip(irreps, irreps.slices(), strict=True):
        block = features[..., block_slice].reshape(-1, multiplicity, irrep.dim)
        blocks.append(block.transpose(1, 2).reshape(-1, multiplicity * irrep.dim))
    return torch.cat(blocks, dim=-1)


def test_ir_mul_to_mul_ir_inverts_the_cuequivariance_layout():
    cuequivariance = pytest.importorskip("cuequivariance")
    cuequivariance_torch = pytest.importorskip("cuequivariance_torch")
    irreps = Irreps("3x0e+2x1o+4x0e+3x1o+2x2e")
    features = torch.randn(5, irreps.dim)

    to_ir_mul = cuequivariance_torch.TransposeIrrepsLayout(
        cuequivariance.Irreps("O3", str(irreps)),
        source=cuequivariance.mul_ir,
        target=cuequivariance.ir_mul,
        use_fallback=True,
    )
    ir_mul_features = to_ir_mul(features)
    assert not torch.equal(ir_mul_features, features)
    torch.testing.assert_close(ir_mul_to_mul_ir(ir_mul_features, irreps), features)


def test_ir_mul_to_mul_ir_restores_e3nn_equivariance():
    irreps = Irreps("2x0e+3x1o")
    features = irreps.randn(4, -1)
    rotation = o3.rand_matrix()
    rotated = features @ irreps.D_from_matrix(rotation).T

    restored = ir_mul_to_mul_ir(_mul_ir_to_ir_mul(features, irreps), irreps)
    restored_rotated = ir_mul_to_mul_ir(_mul_ir_to_ir_mul(rotated, irreps), irreps)
    torch.testing.assert_close(
        restored_rotated, restored @ irreps.D_from_matrix(rotation).T
    )


def test_ir_mul_to_mul_ir_checks_the_width():
    with pytest.raises(ValueError, match="channels"):
        ir_mul_to_mul_ir(torch.zeros(2, 5), Irreps("2x1o"))


def _loop_padding(values, system_index, atom_counts, max_atoms):
    padded = values.new_zeros((atom_counts.shape[0], max_atoms, values.shape[-1]))
    start = 0
    for system, count in enumerate(atom_counts.tolist()):
        padded[system, :count] = values[start : start + count]
        start += count
    return padded


def test_pad_per_system_matches_the_reference_loop():
    atom_counts = torch.tensor([3, 1, 4])
    system_index = torch.repeat_interleave(torch.arange(3), atom_counts)
    values = torch.randn(int(atom_counts.sum()), 5)
    torch.testing.assert_close(
        pad_per_system(values, system_index, atom_counts, max_atoms=6),
        _loop_padding(values, system_index, atom_counts, max_atoms=6),
    )


def test_centre_per_system_subtracts_each_mean():
    atom_counts = torch.tensor([2, 3])
    system_index = torch.repeat_interleave(torch.arange(2), atom_counts)
    positions = torch.randn(5, 3) + torch.tensor([10.0, -4.0, 2.0])
    centred = centre_per_system(positions, system_index, atom_counts)
    torch.testing.assert_close(centred[:2].mean(0), torch.zeros(3))
    torch.testing.assert_close(centred[2:].mean(0), torch.zeros(3))
    torch.testing.assert_close(centred[1] - centred[0], positions[1] - positions[0])


def _water_sample() -> Sample:
    atoms = molecule("H2O")
    return Sample(
        atomic_positions=torch.tensor(atoms.get_positions(), dtype=torch.float32),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers()),
        total_charge=torch.tensor(0.0),
        multiplicity=torch.tensor(1.0),
    )


def test_unsorted_system_index_is_rejected():
    sample = yield_molecules_collate_fn([_water_sample(), _water_sample()])
    assert sample.system_index is not None
    sample.system_index = sample.system_index.flip(0)
    with pytest.raises(ValueError, match="sorted"):
        _sample_to_simstate(sample, r_max=5.0)


@pytest.mark.usefixtures("torch_sim_compiler")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuequivariance needs CUDA")
@pytest.mark.skipif(not MACE_OFF_MEDIUM.exists(), reason="MACE-OFF23 medium missing")
def test_cuequivariance_node_features_match_e3nn():
    """With enable_cueq the MACE features come out in ir_mul layout; the
    preprocessor must transpose them back to the e3nn features."""
    samples = [_water_sample()]
    atoms = molecule("CH3CHO")
    samples.append(
        Sample(
            atomic_positions=torch.tensor(atoms.get_positions(), dtype=torch.float32),
            atomic_numbers=torch.tensor(atoms.get_atomic_numbers()),
            total_charge=torch.tensor(0.0),
            multiplicity=torch.tensor(1.0),
        )
    )
    sample = yield_molecules_collate_fn(samples).to_(torch.device("cuda"))

    features = {}
    for enable_cueq in (False, True):
        config = MaceConfig(
            model_path=MACE_OFF_MEDIUM,
            device="cuda",
            dtype="float32",
            enable_cueq=enable_cueq,
        )
        irreps = config.get_irrep_signature()
        model = config.build_torch_sim_model()
        assert bool(getattr(model, "_enable_cueq", False)) == enable_cueq
        state, _ = _sample_to_simstate(sample, r_max=float(model.r_max))
        with torch.inference_mode():
            features[enable_cueq] = _capture_node_feats(model, state, irreps).cpu()

    torch.testing.assert_close(features[True], features[False], atol=1e-4, rtol=1e-4)
