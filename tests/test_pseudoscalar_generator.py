import numpy.testing as npt
import torch
from e3nn.o3 import Irreps
from mace.calculators import mace_mp

from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig
from threedscriptors.data_handling.data_utils import get_ase_atoms
from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    PseudoscalarGenerator,
)
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

    calc = mace_mp("medium")

    pos = atoms.get_positions()
    atoms2 = atoms.copy()
    pos2 = pos * -1.0
    atoms2.set_positions(pos2)

    des1 = calc.get_descriptors(atoms, invariants_only=False)
    des1 = torch.Tensor(des1).unsqueeze(0)
    des2 = calc.get_descriptors(atoms2, invariants_only=False)
    des2 = torch.Tensor(des2).unsqueeze(0)
    calculator_irreps = get_mace_calculator_irrep_signature(calc)

    print(calculator_irreps)

    embedding_preprocessor_config = EmbeddingPreprocessConfig(
        input_irreps=calculator_irreps,
        pseudoscalars=True,
        pseudoscalar_dimension=128,
        pseudoscalar_embedding_dim=128,
        input_embedding_size=640,
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




#def test_fixed_ps_generator():
#
#    smiles = "CC(N)O"
#    atoms = get_ase_atoms(smiles)
#
#    calc = mace_mp("medium",default_dtype="float64")
#
#    pos = atoms.get_positions()
#    atoms2 = atoms.copy()
#    pos2 = pos * -1.0
#    atoms2.set_positions(pos2)
#
#    des1 = calc.get_descriptors(atoms, invariants_only=False)
#    des1 = torch.Tensor(des1).unsqueeze(0)
#    des2 = calc.get_descriptors(atoms2, invariants_only=False)
#    des2 = torch.Tensor(des2).unsqueeze(0)
#    calculator_irreps = get_mace_calculator_irrep_signature(calc)
#
#    inv_indices, _ = get_invariant_indices(calculator_irreps)
#    _, des1_equiv = split_invariants_equivariants(des1, inv_indices)
#    _, des2_equiv = split_invariants_equivariants(des2, inv_indices)
#
#
#    npt.assert_allclose(des1_equiv, -1* des2_equiv)
#
#    #des1_equiv = torch.rand_like(des1_equiv)
#    #des2_equiv = -1* des1_equiv
#
#
#    embedding_preprocessor_config = EmbeddingPreprocessConfig(
#        input_irreps=calculator_irreps,
#        pseudoscalars=True,
#        pseudoscalar_dimension=128,
#        pseudoscalar_embedding_dim=128,
#        input_embedding_size=640,
#    )
#
#    print(des1_equiv.shape)
#
#
#
#
#    # ⬇︎ accumulate first and second moments of ALL Cartesian comps
#    s1 = torch.zeros(3, dtype=torch.float64)                 # sum of components
#    s2 = torch.zeros(3, dtype=torch.float64)                 # sum of squares
#    n  = 0
#    vecs = des1_equiv[..., :].reshape(-1, 3)   # rows = individual 1o vectors
#
#
#    std = vecs.std(0, unbiased=True)
#    print(std)  # (3,)
#    scale = 1.0 / (std.mean() + 1e-10)             # shape (3,)
#
#
#
#    des1_equiv = des1_equiv * scale
#    des2_equiv = des2_equiv * scale
#
#
#
#    ps_generator = FixedPseudoscalar(embedding_preprocessor_config)
#    #print(sum(p.numel() for p in ps_generator.parameters() if p.requires_grad))
#
#
#    ps_1 = ps_generator(des1_equiv).detach()
#    ps_2 = ps_generator(des2_equiv).detach()
#
#    npt.assert_allclose(ps_1, -1* ps_2)


