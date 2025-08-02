import pytest
import torch
from ase import Atoms
from mace.calculators import mace_mp

from threedscriptors.configuration.architecture_config import (
    RadialBasisFunctionType,
    RelativeDistancePositionalEncodingConfig,
)
from threedscriptors.configuration.data_config import (
    DatasetConfig,
    DatasetTypes,
    MaceCalculatorConfig,
    TaskConfig,
)
from threedscriptors.data_handling.data_utils import get_ase_atoms


@pytest.fixture(scope="session")
def embeddings():
    return torch.rand((5, 5))


@pytest.fixture(scope="session")
def padding_mask():
    return torch.ones((5, 1))


@pytest.fixture(scope="session")
def regression_target():
    return torch.tensor([1.0])


@pytest.fixture(scope="session")
def regression_targets():
    return torch.ones((5, 5))


@pytest.fixture(scope="session")
def regression_masks():
    return torch.ones((5, 5))


@pytest.fixture(scope="session")
def regression_mask():
    return torch.tensor([1])


@pytest.fixture(scope = "session")
def molecule():

    smiles = "C"
    atoms : Atoms =  get_ase_atoms(smiles)
    return atoms



@pytest.fixture(scope="session")
def sample_smiles():
    smiles = [
        "COC1=CC=CC(Cl)=C1NC(=O)N1CCC[C@H](C(N)=O)C1",
        "O=C(NCC(F)F)[C@H](NC1=CC2=C(C=C1Br)CNC2)C1=CC(Cl)=CC(C2CC2)=C1",
        "O=C(NCC(F)F)[C@H](NC1=CC=C2CNCC2=C1)C1=CC(Br)=CC2=C1NC=N2",
        "NC(=O)[C@H]1CCCN(C(=O)CC2=CC=CC3=C2C=CO3)C1",
        "CC1=CC(CC(=O)N2CCC[C@H](C(N)=O)C2)=CC=N1",
    ]

    return smiles


@pytest.fixture(scope="session")
def sample_dataset_config():
    return DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
    )


@pytest.fixture(scope="session")
def sample_chiral_dataset_config():
    return DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        tasks=[TaskConfig(task_name="cmrt")],
    )


@pytest.fixture(scope="session")
def sample_regression_dataset_config():
    return DatasetConfig(
        N_molecules=5,
        dataset_type=DatasetTypes.REGRESSION_DATASET,
        BFGS_tol=0.2,
        BFGS_max_steps=500,
        N_conformers=1,
        embedding_model_config=MaceCalculatorConfig(
            mace_calc=mace_mp("medium"), model_name="medium"
        ),
        tasks=[
            TaskConfig(task_name="HLM"),
            TaskConfig(task_name="LogD"),
            TaskConfig(task_name="KSOL"),
        ],
    )


@pytest.fixture(scope="session")
def positional_encoding_config():

    return RelativeDistancePositionalEncodingConfig(N_radial_basis_functions=16, distance_cutoff=20.0, d_projection=64, basis_function_type= RadialBasisFunctionType.GAUSSIAN)
