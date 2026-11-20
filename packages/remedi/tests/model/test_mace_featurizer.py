"""The MACE featurizer against the full MACE model it is lifted from.

MACE-OFF23 medium on the CPU in float64: the featurizer's node features equal the
model's own (radius graph, readouts and all), do not depend on batching or padding,
and compile into one graph that serves further shapes without recompiling.
MACE-POLAR-1-M runs only with RUN_POLAR_TESTS=1, cuequivariance only on CUDA.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from ase import Atoms
from ase.build import molecule
from e3nn import o3
from e3nn.o3 import Irreps

from remedi.configuration.mace_config import MaceConfig
from remedi.model.featurizer import (
    MaceFeaturizer,
    check_distant_pairs_are_inert,
    ir_mul_to_mul_ir_index,
    molecule_pair_index,
    pad_molecules,
)
from remedi.training.data.sample import Sample, yield_molecules_collate_fn

MACE_OFF_MEDIUM = Path.home() / ".cache" / "mace" / "MACE-OFF23_medium.model"
MACE_POLAR = Path(
    os.environ.get(
        "MACE_POLAR_MODEL_PATH",
        "/dais/u/wedigs/graph_electrostatics/MACE-POLAR-1-M.model",
    )
)
POLAR_TESTS = pytest.mark.skipif(
    os.environ.get("RUN_POLAR_TESTS") != "1" or not MACE_POLAR.exists(),
    reason="set RUN_POLAR_TESTS=1 (and MACE_POLAR_MODEL_PATH) to run MACE-POLAR",
)
CUDA_TESTS = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuequivariance needs CUDA"
)


def _molecules() -> list[Atoms]:
    """Small molecules and C60, whose atoms are up to 7 A apart (beyond the cutoff
    of MACE-OFF, 5 A, and of MACE-POLAR, 6 A)."""
    names = ["H2O", "CH3CH2OH", "C6H6", "CH3COOH", "C60"]
    structures = []
    generator = torch.Generator().manual_seed(0)
    for name in names:
        atoms = molecule(name)
        # Off the symmetric equilibrium frame, so no feature vanishes by symmetry.
        noise = torch.randn(len(atoms), 3, generator=generator, dtype=torch.float64)
        atoms.positions += 0.05 * noise.numpy()
        structures.append(atoms)
    return structures


def _sample(atoms: Atoms) -> Sample:
    return Sample(
        atomic_positions=torch.tensor(atoms.get_positions(), dtype=torch.float64),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers()),
        total_charge=torch.tensor(0.0),
        multiplicity=torch.tensor(1.0),
    )


def _featurize(
    featurizer: MaceFeaturizer, structures: list[Atoms], pad_multiple: int = 1
) -> list[torch.Tensor]:
    """The featurizer's features of each structure, ``(atoms, C)``, on the CPU."""
    batch = yield_molecules_collate_fn([_sample(atoms) for atoms in structures])
    device = next(featurizer.parameters()).device
    batch.to_(device)
    assert batch.atomic_numbers is not None and batch.atomic_positions is not None
    assert batch.system_index is not None and batch.padding_mask is not None
    molecules = pad_molecules(
        batch.atomic_numbers,
        batch.atomic_positions,
        batch.system_index,
        batch.padding_mask,
        pad_multiple,
    )
    features = featurizer(molecules.atomic_numbers, molecules.positions).cpu()
    return [features[index, : len(atoms)] for index, atoms in enumerate(structures)]


def _full_model_features(
    model_path: Path, structures: list[Atoms]
) -> list[torch.Tensor]:
    """The node features the full MACE model returns (its own radius graph), per
    structure, in float64 on the CPU."""
    from mace.calculators import MACECalculator

    calculator = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        default_dtype="float64",
        enable_cueq=False,
    )
    model = calculator.models[0]
    assert isinstance(model, torch.nn.Module)
    captured: list[torch.Tensor] = []
    handle = model.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output["node_feats"].detach())
    )
    try:
        for atoms in structures:
            calculator.get_descriptors(atoms, invariants_only=False)
    finally:
        handle.remove()
    return captured


def _assert_matches(
    features: list[torch.Tensor], references: list[torch.Tensor], tolerance: float
) -> None:
    for feature, reference in zip(features, references, strict=True):
        assert feature.shape == reference.shape
        scale = reference.abs().max()
        assert float((feature - reference).abs().max() / scale) < tolerance


def _config(model_path: Path, **overrides) -> MaceConfig:
    settings = {
        "model_path": model_path,
        "device": "cpu",
        "dtype": "float64",
        "enable_cueq": False,
    }
    settings.update(overrides)
    return MaceConfig.model_validate(settings)


@pytest.fixture(scope="module")
def mace_off_medium() -> Path:
    if not MACE_OFF_MEDIUM.exists():
        pytest.skip(f"{MACE_OFF_MEDIUM} is missing")
    return MACE_OFF_MEDIUM


@pytest.fixture(scope="module")
def structures() -> list[Atoms]:
    return _molecules()


@pytest.fixture(scope="module")
def mace_off_references(mace_off_medium: Path, structures) -> list[torch.Tensor]:
    return _full_model_features(mace_off_medium, structures)


@pytest.fixture(scope="module")
def mace_off_featurizer(mace_off_medium: Path) -> MaceFeaturizer:
    return _config(mace_off_medium).build_featurizer()


# ---------------------------------------------------------------- building blocks


def test_molecule_pair_index_lists_every_ordered_pair_within_each_molecule():
    pairs = molecule_pair_index(molecules=2, atoms=3, device=torch.device("cpu"))
    expected = {
        (molecule_index * 3 + sender, molecule_index * 3 + receiver)
        for molecule_index in range(2)
        for sender in range(3)
        for receiver in range(3)
        if sender != receiver
    }
    listed = [tuple(pair) for pair in pairs.T.tolist()]
    assert len(listed) == len(expected)
    assert set(listed) == expected


def _mul_ir_to_ir_mul(features: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    blocks = []
    for (multiplicity, irrep), block_slice in zip(irreps, irreps.slices(), strict=True):
        block = features[..., block_slice].reshape(-1, multiplicity, irrep.dim)
        blocks.append(block.transpose(1, 2).reshape(-1, multiplicity * irrep.dim))
    return torch.cat(blocks, dim=-1)


def test_ir_mul_index_inverts_the_cuequivariance_layout():
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
    index = ir_mul_to_mul_ir_index(irreps)
    torch.testing.assert_close(ir_mul_features[..., index], features)


def test_ir_mul_index_restores_e3nn_equivariance():
    irreps = Irreps("2x0e+3x1o")
    features = irreps.randn(4, -1)
    rotation = o3.rand_matrix()
    rotated = features @ irreps.D_from_matrix(rotation).T
    index = ir_mul_to_mul_ir_index(irreps)
    restored = _mul_ir_to_ir_mul(features, irreps)[..., index]
    restored_rotated = _mul_ir_to_ir_mul(rotated, irreps)[..., index]
    torch.testing.assert_close(
        restored_rotated, restored @ irreps.D_from_matrix(rotation).T
    )


def test_pad_molecules_places_and_centres_each_molecule():
    structures = [molecule("H2O"), molecule("CH4")]
    batch = yield_molecules_collate_fn([_sample(atoms) for atoms in structures])
    assert batch.atomic_numbers is not None and batch.atomic_positions is not None
    assert batch.system_index is not None and batch.padding_mask is not None
    molecules = pad_molecules(
        batch.atomic_numbers,
        batch.atomic_positions,
        batch.system_index,
        batch.padding_mask,
        pad_multiple=4,
    )
    assert molecules.atomic_numbers.shape == (2, 8)
    assert molecules.atom_counts.tolist() == [3, 5]
    for index, atoms in enumerate(structures):
        count = len(atoms)
        assert molecules.atomic_numbers[index, :count].tolist() == list(atoms.numbers)
        assert not molecules.atomic_numbers[index, count:].any()
        assert molecules.padding_mask[index, count:].all()
        assert not molecules.padding_mask[index, :count].any()
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float64)
        torch.testing.assert_close(
            molecules.positions[index, :count], positions - positions.mean(dim=0)
        )
        assert not molecules.positions[index, count:].any()


def test_pad_molecules_gives_single_atoms_two_slots():
    batch = yield_molecules_collate_fn([_sample(Atoms("Ne", positions=[[1.0, 2, 3]]))])
    assert batch.atomic_numbers is not None and batch.atomic_positions is not None
    assert batch.system_index is not None and batch.padding_mask is not None
    molecules = pad_molecules(
        batch.atomic_numbers,
        batch.atomic_positions,
        batch.system_index,
        batch.padding_mask,
    )
    assert molecules.padding_mask.tolist() == [[False, True]]


def test_unsorted_system_index_is_rejected():
    batch = yield_molecules_collate_fn([_sample(molecule("H2O"))] * 2)
    assert batch.atomic_numbers is not None and batch.atomic_positions is not None
    assert batch.system_index is not None and batch.padding_mask is not None
    with pytest.raises(RuntimeError, match="sorted"):
        pad_molecules(
            batch.atomic_numbers,
            batch.atomic_positions,
            batch.system_index.flip(0),
            batch.padding_mask,
        )


# ---------------------------------------------------------------- MACE-OFF, CPU


def test_featurizer_matches_the_full_model(
    mace_off_featurizer, mace_off_references, structures
):
    distances = torch.cdist(*(2 * [torch.tensor(structures[-1].positions)]))
    assert (distances > mace_off_featurizer.r_max).any()
    features = _featurize(mace_off_featurizer, structures)
    _assert_matches(features, mace_off_references, tolerance=1e-12)


def test_featurizer_does_not_depend_on_batching_or_padding(
    mace_off_featurizer, structures
):
    batched = _featurize(mace_off_featurizer, structures, pad_multiple=8)
    for atoms, in_batch in zip(structures, batched, strict=True):
        alone = _featurize(mace_off_featurizer, [atoms])[0]
        torch.testing.assert_close(in_batch, alone, rtol=1e-12, atol=1e-12)


def test_featurizer_rejects_unknown_elements(mace_off_featurizer):
    atomic_numbers = torch.tensor([[6, 2]])  # helium: not a MACE-OFF element
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]])
    with pytest.raises(RuntimeError, match="element"):
        mace_off_featurizer(atomic_numbers, positions)


def test_featurizer_rejects_models_whose_distant_pairs_contribute(mace_off_medium):
    model = _config(mace_off_medium)._load_raw_model()
    check_distant_pairs_are_inert(model)
    biased = type(model.interactions[0].conv_tp_weights)
    original = model.interactions[0].conv_tp_weights
    try:
        # A radial network with a bias gives pairs beyond the cutoff nonzero weights.
        model.interactions[0].conv_tp_weights = torch.nn.Linear(8, 512)
        with pytest.raises(ValueError, match="vanish beyond the cutoff"):
            check_distant_pairs_are_inert(model)
    finally:
        model.interactions[0].conv_tp_weights = original
    assert type(model.interactions[0].conv_tp_weights) is biased


@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_featurizer_is_one_graph_for_every_shape(
    mace_off_medium, mace_off_references, structures
):
    featurizer = _config(mace_off_medium).build_featurizer()
    featurizer.compile_features()
    torch._dynamo.reset()
    with torch._dynamo.config.patch(
        recompile_limit=1, fail_on_recompile_limit_hit=True
    ):
        _assert_matches(
            _featurize(featurizer, structures), mace_off_references, tolerance=1e-12
        )
        # Fewer molecules, other atom slots: the same graph.
        _assert_matches(
            _featurize(featurizer, structures[1:4], pad_multiple=3),
            mace_off_references[1:4],
            tolerance=1e-12,
        )


# ---------------------------------------------------------------- MACE-POLAR


@POLAR_TESTS
def test_polar_featurizer_matches_the_full_model(structures):
    references = _full_model_features(MACE_POLAR, structures)
    featurizer = _config(MACE_POLAR).build_featurizer()
    assert featurizer.permute_edge_vectors
    _assert_matches(_featurize(featurizer, structures), references, tolerance=1e-12)


# ---------------------------------------------------------------- cuequivariance


def _cueq_against_e3nn(model_path: Path, structures: list[Atoms]) -> None:
    e3nn_features = _featurize(
        _config(model_path, device="cuda", dtype="float32").build_featurizer(),
        structures,
    )
    featurizer = _config(
        model_path, device="cuda", dtype="float32", enable_cueq=True
    ).build_featurizer()
    featurizer.compile_features()
    assert featurizer.layout == "ir_mul"
    torch._dynamo.reset()
    with torch._dynamo.config.patch(
        recompile_limit=1, fail_on_recompile_limit_hit=True
    ):
        cueq_features = _featurize(featurizer, structures)
        _featurize(featurizer, structures[1:4], pad_multiple=3)
    _assert_matches(cueq_features, e3nn_features, tolerance=1e-4)


@CUDA_TESTS
@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_cueq_featurizer_matches_e3nn(mace_off_medium, structures):
    _cueq_against_e3nn(mace_off_medium, structures)


@CUDA_TESTS
@POLAR_TESTS
@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_cueq_polar_featurizer_matches_e3nn(structures):
    _cueq_against_e3nn(MACE_POLAR, structures)
