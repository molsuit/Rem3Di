import numpy as np
import numpy.testing as npt
import torch
from e3nn.o3 import Irreps
from mace.calculators import MACECalculator, mace_mp

from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig
from threedscriptors.configuration.data_config import DatasetConfig, DatasetTypes
from threedscriptors.data_handling.cmrt_preprocessing import load_cmrt_data
from threedscriptors.data_handling.data_utils import get_ase_atoms
from threedscriptors.data_handling.dataset_builder import (
    DatasetBuildingDirector,
)
from threedscriptors.data_handling.smiles_iterator import ListSmilesIterator
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

    embedding_preprocessor_config = EmbeddingPreprocessConfig(
        input_irreps=calculator_irreps, pseudoscalars=True, pseudoscalar_dimension=128
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


def test_chiral_dataset():
    smiles, regression_targets, regression_masks, aux_data, tasks = load_cmrt_data()

    MACE_PATH = (
        "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
    )
    # Get the train and test data-loaders

    dataset_config = DatasetConfig(
        N_molecules=2,
        dataset_type=DatasetTypes.REGRESSION,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model=MACE_PATH,
        max_atoms=None,
        tasks=tasks,
    )
    iterator = ListSmilesIterator(smiles)
    db_director, dataset = DatasetBuildingDirector.build_chiral_dataset(
        iterator=iterator,
        dataset_config=dataset_config,
        regression_targets=regression_targets,
        regression_masks=regression_masks,
        auxillary_data=aux_data,
        return_normalized_targets=False,
    )

    print(db_director.builder.index_list)
    print(dataset.regression_targets)
    mol_0 = dataset.molecules[0]

    mol_1 = dataset.molecules[1]

    print(mol_1.get_positions())
    print(mol_0.get_positions())

    assert (mol_0.get_positions() == -mol_1.get_positions()).all()

    mace_calc = MACECalculator(MACE_PATH, device="cuda", enable_cueq=True)

    input_irreps = get_mace_calculator_irrep_signature(mace_calc)

    mol_0_embedding = torch.Tensor(
        mace_calc.get_descriptors(mol_0, invariants_only=False)
    )
    mol_1_embedding = torch.Tensor(
        mace_calc.get_descriptors(mol_1, invariants_only=False)
    )

    print(mol_0_embedding.shape)

    embedding_preprocessor_config = EmbeddingPreprocessConfig(
        pseudoscalars=True, pseudoscalar_dimension=8, input_irreps=input_irreps
    )

    ps_generator = PseudoscalarGenerator(embedding_preprocessor_config)

    print(sum(p.numel() for p in ps_generator.parameters() if p.requires_grad))

    pseudoscalars_0 = ps_generator.get_pseudoscalars(mol_0_embedding)
    pseudoscalars_1 = ps_generator.get_pseudoscalars(mol_1_embedding)

    print(pseudoscalars_0)
    print(pseudoscalars_1)

    print(pseudoscalars_1 + pseudoscalars_0)

    assert np.allclose(
        pseudoscalars_0.detach().numpy(),
        -1 * pseudoscalars_1.detach().numpy(),
        rtol=0.1,
    )


test_chiral_dataset()
