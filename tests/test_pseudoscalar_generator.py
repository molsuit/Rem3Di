import numpy.testing as npt
import torch
from e3nn.o3 import Irreps
from mace.calculators import MACECalculator

from threedscriptors.data_handling.preprocessing import get_ase_atoms
from threedscriptors.model.architecture_config import EmbeddingPreprocessConfig
from threedscriptors.model.atomic_descriptor_preprocess import PseudoscalarGenerator
from threedscriptors.utils.model_utils import (
    get_invariant_indices,
    get_mace_calculator_irrep_signature,
    get_pseudoscalars_indices,
)


def test_get_pseudoscalars():
    x = Irreps("10x0e+10x0o")

    indices, out_irreps = get_pseudoscalars_indices(x)
    assert indices == [i for i in range(10, 20)]
    assert out_irreps == Irreps("10x0o")


def test_pseudoscalar_generator():
    smiles = "CC(N)O"
    atoms = get_ase_atoms(smiles)

    calc = MACECalculator(
        model_paths="/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model",
        device="cpu",
    )

    pos = atoms.get_positions()
    atoms2 = atoms.copy()
    pos2 = pos * -1.0
    atoms2.set_positions(pos2)

    des1 = calc.get_descriptors(atoms, invariants_only=False)
    des1 = torch.Tensor(des1).unsqueeze(0)
    des2 = calc.get_descriptors(atoms2, invariants_only=False)
    des2 = torch.Tensor(des2).unsqueeze(0)
    calculator_irreps = get_mace_calculator_irrep_signature(calc)

    embedding_preprocessor_config = EmbeddingPreprocessConfig(
        input_irreps=calculator_irreps, pseudoscalars=True
    )
    ps_generator = PseudoscalarGenerator(embedding_preprocessor_config)

    ps_indices, _ = get_pseudoscalars_indices(ps_generator.config.output_irreps)

    des_ps = ps_generator.forward(des1)
    des_ps2 = ps_generator.forward(des2)

    npt.assert_array_almost_equal(
        des_ps[:, :, ps_indices].detach().numpy(),
        -des_ps2[:, :, ps_indices].detach().numpy(),
    )


def test_get_invariant_indices():
    x = Irreps("10x0e+10x1o+10x0e")
    _, out_irreps = get_invariant_indices(x)

    assert out_irreps == Irreps("10x0e+10x0e")
