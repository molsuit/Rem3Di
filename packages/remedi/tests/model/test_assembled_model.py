"""End-to-end checks of the assembled Rem3Di encoder with on-the-fly MACE.

Uses MACE-OFF23 medium on the CPU: the descriptor is rotation invariant, the
chiral (pseudoscalar) embedding flips sign under a mirror image, and a
molecule's descriptor does not depend on what it is batched with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms
from ase.build import molecule
from scipy.spatial.transform import Rotation

from remedi.configuration.architecture_config import EncoderOnlyArchitectureConfig
from remedi.embedding.calculator import RemediCalculator
from remedi.model.checkpoint import ARCHITECTURE_FILENAME, save_checkpoint
from remedi.training.data.sample import Sample, yield_molecules_collate_fn
from remedi_data.yaml_io import yaml_text

MACE_OFF_MEDIUM = Path.home() / ".cache" / "mace" / "MACE-OFF23_medium.model"

# CHFClBr: a single stereocentre, so its mirror image is the other enantiomer.
CHIRAL_MOLECULE = Atoms(
    "CHFClBr",
    positions=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.03, 0.0, -0.36],
        [-0.51, -0.89, -0.6],
        [-0.52, 0.95, -0.66],
    ],
)


def _architecture(
    model_path: Path, dtype: str = "float32"
) -> EncoderOnlyArchitectureConfig:
    return EncoderOnlyArchitectureConfig.model_validate(
        {
            "mace_config": {
                "model_path": str(model_path),
                "device": "cpu",
                "dtype": dtype,
                "enable_cueq": False,
            },
            "embedding_preprocess_config": {
                "pseudoscalars": True,
                "pseudoscalar_dimension": 8,
                "chiral_embedding_dimension": 8,
                "mace_layer_indices": [0],
                "invariant_normalization_config": {
                    "kind": "on_the_fly_normalization",
                    "momentum": 0.1,
                    "warm_up_batches": 1,
                },
            },
            "encoder_config": {
                "N_layers": 2,
                "attention_layer_config": {
                    "num_heads": 2,
                    "dim_feedforward": 32,
                    "dropout": 0.0,
                },
            },
            "global_aggregator_config": {
                "aggregator_type_config": {
                    "aggregator_type": "pma_attention",
                    "head_dim": 4,
                    "num_heads": 2,
                    "num_seeds": 3,
                },
                "output_dim": 16,
            },
            "positional_encoding_config": {
                "N_radial_basis_functions": 8,
                "distance_cutoff": 5.0,
                "d_projection": 8,
            },
        }
    )


@pytest.fixture(scope="module")
def mace_off_medium() -> Path:
    if not MACE_OFF_MEDIUM.exists():
        pytest.skip(f"{MACE_OFF_MEDIUM} is missing")
    return MACE_OFF_MEDIUM


@pytest.fixture(scope="module")
def architecture(mace_off_medium: Path) -> EncoderOnlyArchitectureConfig:
    return _architecture(mace_off_medium)


@pytest.fixture(scope="module")
def model(architecture):
    torch.manual_seed(0)
    built = architecture.build()
    # The pair bias and the pair-FFN gate are zero-initialised; randomise them
    # so the pair representation actually reaches the descriptor.
    for module in built.modules():
        if hasattr(module, "pair2bias"):
            torch.nn.init.normal_(module.pair2bias.weight)
        if isinstance(getattr(module, "gate", None), torch.nn.Linear):
            torch.nn.init.normal_(module.gate.weight)
    return built.eval()


def _sample(atoms: Atoms, positions: np.ndarray | None = None) -> Sample:
    coordinates = atoms.get_positions() if positions is None else positions
    return Sample(
        atomic_positions=torch.tensor(coordinates, dtype=torch.float32),
        atomic_numbers=torch.tensor(atoms.get_atomic_numbers()),
        total_charge=torch.tensor(0.0),
        multiplicity=torch.tensor(1.0),
    )


def _descriptors(model, samples: list[Sample]) -> torch.Tensor:
    with torch.no_grad():
        return model(yield_molecules_collate_fn(samples)).molecular_descriptor.flat


def _chiral_embeddings(model, samples: list[Sample]) -> torch.Tensor:
    with torch.no_grad():
        preprocessed = model.preprocessor(yield_molecules_collate_fn(samples))
    assert preprocessed.chiral_embeddings is not None
    return preprocessed.chiral_embeddings


@pytest.mark.usefixtures("torch_sim_compiler")
def test_descriptor_is_rotation_invariant(model):
    positions = CHIRAL_MOLECULE.get_positions()
    rotation = Rotation.random(random_state=1).as_matrix()
    reference = _descriptors(model, [_sample(CHIRAL_MOLECULE)])
    rotated = _descriptors(model, [_sample(CHIRAL_MOLECULE, positions @ rotation.T)])
    assert reference.abs().max() > 1e-2
    torch.testing.assert_close(rotated, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.usefixtures("torch_sim_compiler")
def test_mirror_image_flips_the_chiral_embedding(model):
    positions = CHIRAL_MOLECULE.get_positions()
    mirrored = positions * np.array([1.0, 1.0, -1.0])
    original = _chiral_embeddings(model, [_sample(CHIRAL_MOLECULE)])[0]
    mirror = _chiral_embeddings(model, [_sample(CHIRAL_MOLECULE, mirrored)])[0]
    assert original.abs().max() > 1e-3
    torch.testing.assert_close(mirror, -original, atol=1e-5, rtol=1e-4)
    # The descriptor tells the enantiomers apart.
    assert not torch.allclose(
        _descriptors(model, [_sample(CHIRAL_MOLECULE, mirrored)]),
        _descriptors(model, [_sample(CHIRAL_MOLECULE)]),
        atol=1e-4,
    )


@pytest.mark.usefixtures("torch_sim_compiler")
def test_descriptor_is_batch_independent(model):
    # C60 pads the batch beyond 25 atoms (where cdist's matmul path kicked in)
    # and its coordinates are far from those of the small molecule.
    alone = _descriptors(model, [_sample(CHIRAL_MOLECULE)])[0]
    water = molecule("H2O")
    buckyball = molecule("C60")
    buckyball.translate([40.0, -30.0, 50.0])
    after_small = _descriptors(model, [_sample(water), _sample(CHIRAL_MOLECULE)])[1]
    before_large = _descriptors(model, [_sample(CHIRAL_MOLECULE), _sample(buckyball)])[
        0
    ]
    torch.testing.assert_close(after_small, alone, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(before_large, alone, atol=1e-5, rtol=1e-4)


@pytest.mark.usefixtures("torch_sim_compiler")
def test_cuequivariance_layout_is_transposed_back(architecture, model, monkeypatch):
    """Simulate a cueq-converted MACE (node features in ir_mul layout) on the
    CPU: the descriptor must be unchanged once the preprocessor transposes the
    features back to mul_ir."""
    reference = _descriptors(model, [_sample(CHIRAL_MOLECULE)])
    mace = model.preprocessor.torch_sim_mace_model
    irreps = architecture.embedding_preprocess_config.input_irreps

    def to_ir_mul(_module, _inputs, output):
        blocks = []
        for (multiplicity, irrep), block_slice in zip(
            irreps, irreps.slices(), strict=True
        ):
            block = output["node_feats"][..., block_slice]
            block = block.reshape(-1, multiplicity, irrep.dim).transpose(1, 2)
            blocks.append(block.reshape(-1, multiplicity * irrep.dim))
        return {**output, "node_feats": torch.cat(blocks, dim=-1)}

    monkeypatch.setattr(mace, "_enable_cueq", True)
    handle = mace.model.register_forward_hook(to_ir_mul)
    try:
        simulated = _descriptors(model, [_sample(CHIRAL_MOLECULE)])
    finally:
        handle.remove()
    torch.testing.assert_close(simulated, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.usefixtures("torch_sim_compiler")
def test_float64_mace_runs(mace_off_medium: Path):
    # Regression: build() cast the whole model with .float(), including the
    # torch-sim MACE wrapper whose own dtype stayed float64 -> dtype mismatch.
    model = _architecture(mace_off_medium, dtype="float64").build().eval()
    mace = model.preprocessor.torch_sim_mace_model
    assert next(mace.model.parameters()).dtype == torch.float64
    descriptors = _descriptors(
        model, [_sample(molecule("CH3CHO")), _sample(molecule("H2O"))]
    )
    assert descriptors.shape == (2, 3 * 16)
    assert torch.isfinite(descriptors).all()


@pytest.mark.usefixtures("torch_sim_compiler")
def test_calculator_runs_mace_on_the_requested_device(
    tmp_path, architecture, model, mace_off_medium: Path
):
    # Regression: RemediConfig.device never reached MACE (built on the yaml's
    # training device), and the yaml's machine-specific MACE path was loaded
    # before the mace_model_path override was applied.
    data = architecture.model_dump(mode="json")
    data["mace_config"]["device"] = "cuda"
    data["mace_config"]["enable_cueq"] = True
    data["mace_config"]["model_path"] = "/nonexistent/MACE-OFF23_medium.model"
    (tmp_path / ARCHITECTURE_FILENAME).write_text(yaml.safe_dump(data))
    save_checkpoint(model, tmp_path)

    calculator = RemediCalculator(
        model_dir=tmp_path, device="cpu", mace_model_path=mace_off_medium
    )
    embedded = calculator.embed_atoms([CHIRAL_MOLECULE])

    assert calculator._model is not None
    mace = calculator._model.preprocessor.torch_sim_mace_model
    assert next(mace.model.parameters()).device.type == "cpu"
    assert not getattr(mace, "_enable_cueq", False)
    np.testing.assert_allclose(
        embedded, _descriptors(model, [_sample(CHIRAL_MOLECULE)]).numpy(), atol=1e-5
    )


def test_architecture_yaml_roundtrip_is_loadable(architecture):
    reloaded = EncoderOnlyArchitectureConfig.model_validate(
        yaml.safe_load(yaml_text(architecture.model_dump(mode="json")))
    )
    assert reloaded.dimensions == architecture.dimensions
