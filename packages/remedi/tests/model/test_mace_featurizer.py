"""The MACE featurizer against the full MACE model it is lifted from.

MACE-OFF23 medium on the CPU in float64: the featurizer's node features on the
collated neighbour list equal the model's own (radius graph, readouts and all), also
for atoms without neighbours and molecules with pairs beyond the cutoff, do not
depend on batching or padding, and compile into one graph that serves further
shapes and pair counts without recompiling.
MACE-POLAR-1-M runs only with RUN_POLAR_TESTS=1, cuequivariance only on CUDA.
"""

from __future__ import annotations

import copy
import importlib.util
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
    pad_molecules,
)
from remedi.training.data.neighbor_list import molecule_neighbor_list
from remedi.training.data.sample import MoleculeCollator, Sample

MACE_OFF_MEDIUM = Path.home() / ".cache" / "mace" / "MACE-OFF23_medium.model"
MACE_POLAR = Path(
    os.environ.get(
        "MACE_POLAR_MODEL_PATH",
        "/dais/u/wedigs/graph_electrostatics/MACE-POLAR-1-M.model",
    )
)
NEEDS_CUEQUIVARIANCE = pytest.mark.skipif(
    importlib.util.find_spec("cuequivariance_torch") is None,
    reason="needs cuequivariance (the remedi[cuda] extra)",
)
POLAR_TESTS = pytest.mark.skipif(
    os.environ.get("RUN_POLAR_TESTS") != "1" or not MACE_POLAR.exists(),
    reason="set RUN_POLAR_TESTS=1 (and MACE_POLAR_MODEL_PATH) to run MACE-POLAR",
)


# A peptide of 79 atoms (with hydrogens), embedded by RDKit: a large molecule.
LARGE_MOLECULE_SMILES = (
    "NCC(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(C)C)C(=O)O"
)


def _large_molecule() -> Atoms:
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom

    rdkit_molecule = Chem.AddHs(Chem.MolFromSmiles(LARGE_MOLECULE_SMILES))
    if rdDistGeom.EmbedMolecule(rdkit_molecule, randomSeed=0) != 0:
        raise RuntimeError("RDKit could not embed the large test molecule")
    return Atoms(
        numbers=[atom.GetAtomicNum() for atom in rdkit_molecule.GetAtoms()],
        positions=rdkit_molecule.GetConformer().GetPositions(),
    )


def _molecules() -> list[Atoms]:
    """Small molecules, C60 and a peptide, whose atoms are up to 7 A and more apart
    (beyond the cutoff of MACE-OFF, 5 A, and of MACE-POLAR, 6 A), a lone atom, and
    two waters 12 A apart (atoms without neighbours, pairs beyond the cutoff)."""
    names = ["H2O", "CH3CH2OH", "C6H6", "CH3COOH", "C60"]
    structures = [molecule(name) for name in names]
    structures.append(_large_molecule())
    structures.append(Atoms("O", positions=[[0.3, -0.2, 0.1]]))
    distant_waters = molecule("H2O") + molecule("H2O")
    distant_waters.positions[3:] += [12.0, 0.0, 0.0]
    structures.append(distant_waters)
    generator = torch.Generator().manual_seed(0)
    for atoms in structures:
        # Off the symmetric equilibrium frame, so no feature vanishes by symmetry.
        noise = torch.randn(len(atoms), 3, generator=generator, dtype=torch.float64)
        atoms.positions += 0.05 * noise.numpy()
    return structures


def _sample(atoms: Atoms) -> Sample:
    return Sample(
        atomic_positions=torch.tensor(atoms.get_positions(), dtype=torch.float64),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers()),
        total_charge=torch.tensor(0.0),
        multiplicity=torch.tensor(1.0),
    )


def _collate(featurizer: MaceFeaturizer, structures: list[Atoms]) -> Sample:
    return MoleculeCollator(cutoff=featurizer.r_max)(
        [_sample(atoms) for atoms in structures]
    )


def _featurize(
    featurizer: MaceFeaturizer, structures: list[Atoms], pad_multiple: int = 1
) -> list[torch.Tensor]:
    """The featurizer's features of each structure, ``(atoms, C)``, on the CPU."""
    batch = _collate(featurizer, structures)
    batch.to_(next(featurizer.parameters()).device)
    features = featurizer(pad_molecules(batch, pad_multiple)).cpu()
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
        if scale == 0:  # a lone atom of a model without residual first layer
            assert torch.equal(feature, reference)
            continue
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


def test_neighbor_list_holds_the_pairs_within_the_cutoff(structures):
    cutoff = 5.0
    positions = [torch.tensor(atoms.positions) for atoms in structures]
    pairs = molecule_neighbor_list(positions, cutoff)
    expected = set()
    first_atom = 0
    for molecule_positions in positions:
        distances = torch.cdist(molecule_positions, molecule_positions)
        for sender in range(len(molecule_positions)):
            for receiver in range(len(molecule_positions)):
                if sender != receiver and distances[sender, receiver] < cutoff:
                    expected.add((first_atom + sender, first_atom + receiver))
        first_atom += len(molecule_positions)
    listed = [tuple(pair) for pair in pairs.T.tolist()]
    assert pairs.dtype == torch.long
    assert len(listed) == len(expected)
    assert set(listed) == expected
    # Some pairs are beyond the cutoff (C60, the peptide, the distant waters).
    assert len(listed) < sum(len(p) * (len(p) - 1) for p in positions)


def test_neighbor_list_of_atoms_without_neighbors_is_empty():
    lone_atoms = [torch.zeros((1, 3)), torch.tensor([[0.0, 0, 0], [9.0, 0, 0]])]
    assert molecule_neighbor_list(lone_atoms, 5.0).shape == (2, 0)
    with pytest.raises(ValueError, match="positive"):
        molecule_neighbor_list(lone_atoms, 0.0)


def test_collator_records_its_cutoff_and_rejects_a_bad_one():
    batch = MoleculeCollator(cutoff=4.0)([_sample(molecule("H2O"))])
    assert batch.neighbor_cutoff == 4.0
    assert batch.edge_index is not None and batch.edge_index.shape == (2, 6)
    with pytest.raises(ValueError, match="positive"):
        MoleculeCollator(cutoff=float("nan"))


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
    batch = MoleculeCollator(cutoff=5.0)([_sample(atoms) for atoms in structures])
    molecules = pad_molecules(batch, pad_multiple=4)
    assert molecules.atomic_numbers.shape == (2, 8)
    assert molecules.atom_counts.tolist() == [3, 5]
    assert molecules.neighbor_cutoff == 5.0
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


class _PadMolecules(torch.nn.Module):
    """``pad_molecules`` on plain tensors, as an exported model calls it."""

    def forward(self, atomic_numbers, positions, system_index, padding_mask, edges):
        sample = Sample(
            atomic_positions=positions,
            atomic_numbers=atomic_numbers,
            system_index=system_index,
            padding_mask=padding_mask,
            edge_index=edges,
            neighbor_cutoff=5.0,
        )
        molecules = pad_molecules(sample)
        return molecules.positions, molecules.edge_index


def _flat(batch: Sample) -> tuple[torch.Tensor, ...]:
    fields = (
        batch.atomic_numbers,
        batch.atomic_positions,
        batch.system_index,
        batch.padding_mask,
        batch.edge_index,
    )
    assert all(field is not None for field in fields)
    return tuple(field for field in fields if field is not None)


def test_pad_molecules_exports_with_dynamic_shapes():
    """Regression: math.ceil and max() in the padding specialised the molecule and
    atom-slot counts, so torch.export could not keep them dynamic."""
    collate = MoleculeCollator(cutoff=5.0)
    small = _flat(collate([_sample(molecule(name)) for name in ("H2O", "CH4")]))
    large = _flat(
        collate([_sample(molecule(name)) for name in ("C2H6", "CH3OH", "H2O")])
    )
    atoms = torch.export.Dim("atoms", min=3)
    dynamic_shapes = (
        {0: atoms},
        {0: atoms},
        {0: atoms},
        {0: torch.export.Dim("molecules", min=2), 1: torch.export.Dim("slots", min=2)},
        {1: torch.export.Dim("pairs", min=2)},
    )
    exported = torch.export.export(
        _PadMolecules(), small, dynamic_shapes=dynamic_shapes, strict=False
    )
    for flat_batch in (small, large):
        for result, expected in zip(
            exported.module()(*flat_batch), _PadMolecules()(*flat_batch), strict=True
        ):
            torch.testing.assert_close(result, expected)


def test_pad_molecules_places_the_neighbor_list_on_the_padded_atoms(structures):
    batch = MoleculeCollator(cutoff=5.0)([_sample(atoms) for atoms in structures])
    assert batch.edge_index is not None and batch.atomic_positions is not None
    molecules = pad_molecules(batch, pad_multiple=8)
    atom_slots = molecules.atomic_numbers.shape[1]
    assert molecules.edge_index.shape == batch.edge_index.shape
    flat_vectors = (
        batch.atomic_positions[batch.edge_index[1]]
        - batch.atomic_positions[batch.edge_index[0]]
    )
    padded_positions = molecules.positions.reshape(-1, 3)
    sender, receiver = molecules.edge_index
    torch.testing.assert_close(
        padded_positions[receiver] - padded_positions[sender], flat_vectors
    )
    assert torch.equal(sender // atom_slots, receiver // atom_slots)
    assert not molecules.padding_mask.reshape(-1)[sender].any()
    assert not molecules.padding_mask.reshape(-1)[receiver].any()


def test_pad_molecules_gives_single_atoms_two_slots():
    lone_atom = _sample(Atoms("Ne", positions=[[1.0, 2, 3]]))
    molecules = pad_molecules(MoleculeCollator(cutoff=5.0)([lone_atom]))
    assert molecules.padding_mask.tolist() == [[False, True]]
    assert molecules.edge_index.shape == (2, 0)


def test_pad_molecules_needs_the_neighbor_list():
    batch = MoleculeCollator(cutoff=5.0)([_sample(molecule("H2O"))])
    batch.edge_index = None
    with pytest.raises(ValueError, match="neighbour list"):
        pad_molecules(batch)


def test_unsorted_system_index_is_rejected():
    batch = MoleculeCollator(cutoff=5.0)([_sample(molecule("H2O"))] * 2)
    assert batch.system_index is not None
    batch.system_index = batch.system_index.flip(0)
    with pytest.raises(RuntimeError, match="sorted"):
        pad_molecules(batch)


# ---------------------------------------------------------------- MACE-OFF, CPU


def test_featurizer_matches_the_full_model(
    mace_off_featurizer, mace_off_references, structures
):
    for atoms in structures[-4:]:  # C60, the peptide, the distant waters
        if len(atoms) > 1:
            distances = torch.cdist(*(2 * [torch.tensor(atoms.positions)]))
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


def test_featurizer_stays_frozen_when_its_owner_trains(mace_off_featurizer):
    owner = torch.nn.Sequential(mace_off_featurizer, torch.nn.Linear(2, 2))
    owner.train()
    assert not any(module.training for module in mace_off_featurizer.modules())
    assert owner[1].training
    assert not any(
        parameter.requires_grad for parameter in mace_off_featurizer.parameters()
    )


def test_featurizer_stays_in_float32_under_autocast(mace_off_featurizer, structures):
    """The trained modules may run under bfloat16 autocast; MACE never does."""
    padded = pad_molecules(_collate(mace_off_featurizer, structures[:2]))
    expected = mace_off_featurizer(padded)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        features = mace_off_featurizer(padded)
    assert features.dtype == expected.dtype == mace_off_featurizer.dtype
    torch.testing.assert_close(features, expected, rtol=0.0, atol=0.0)


def test_float32_featurizer_refuses_tf32_matmuls(mace_off_featurizer, structures):
    featurizer = copy.deepcopy(mace_off_featurizer).float()
    padded = pad_molecules(_collate(featurizer, structures[:1]))
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with pytest.raises(RuntimeError, match="full float32 matmuls"):
            featurizer(padded)
    finally:
        torch.set_float32_matmul_precision(previous)


def test_featurizer_rejects_unknown_elements(mace_off_featurizer):
    helium = Atoms("CHe", positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    batch = _collate(mace_off_featurizer, [helium])
    with pytest.raises(RuntimeError, match="element"):
        mace_off_featurizer(pad_molecules(batch))


def test_featurizer_rejects_a_neighbor_list_of_another_cutoff(
    mace_off_featurizer, structures
):
    batch = MoleculeCollator(cutoff=mace_off_featurizer.r_max - 1.0)(
        [_sample(atoms) for atoms in structures]
    )
    with pytest.raises(ValueError, match="cutoff"):
        mace_off_featurizer(pad_molecules(batch))


def test_featurizer_rejects_pairs_across_molecules(mace_off_featurizer):
    batch = _collate(mace_off_featurizer, [molecule("H2O"), molecule("H2O")])
    batch.edge_index = torch.tensor([[0], [3]])
    with pytest.raises(RuntimeError, match="different molecules"):
        mace_off_featurizer(pad_molecules(batch))


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


@pytest.mark.slow
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
        # Fewer molecules, other atom slots and pair counts: the same graph.
        _assert_matches(
            _featurize(featurizer, structures[1:4], pad_multiple=3),
            mace_off_references[1:4],
            tolerance=1e-12,
        )
        # No pairs at all (lone atoms): inert pairs fill in, the same graph.
        lone_atom = structures[-2]
        _assert_matches(
            _featurize(featurizer, [lone_atom, lone_atom]),
            [mace_off_references[-2]] * 2,
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
        _featurize(featurizer, [structures[-2]] * 2)
    _assert_matches(cueq_features, e3nn_features, tolerance=1e-4)


@pytest.mark.gpu
@NEEDS_CUEQUIVARIANCE
@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_cueq_featurizer_matches_e3nn(mace_off_medium, structures):
    _cueq_against_e3nn(mace_off_medium, structures)


@pytest.mark.gpu
@NEEDS_CUEQUIVARIANCE
@POLAR_TESTS
@pytest.mark.usefixtures("inductor_compiler")
def test_compiled_cueq_polar_featurizer_matches_e3nn(structures):
    _cueq_against_e3nn(MACE_POLAR, structures)
