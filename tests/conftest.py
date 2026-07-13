import pytest
import torch
from ase import Atoms

# e3nn 0.4.4 loads o3/constants.pt via torch.load; torch>=2.6 defaults
# weights_only=True and rejects the pickled `slice` objects, so `from e3nn import o3`
# (transitively imported by everything below) raises UnpicklingError. Allowlist it.
# ponytail: drop this once e3nn is upgraded past the torch.load(weights_only) break.
torch.serialization.add_safe_globals([slice])

from remedi.configuration.architecture_config import (  # noqa: E402
    GaussianBasisConfig,
    RelativeDistancePositionalEncodingConfig,
)
from remedi.data_handling.data_utils import get_ase_atoms  # noqa: E402


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


@pytest.fixture(scope="session")
def molecule():
    smiles = "C"
    atoms: Atoms = get_ase_atoms(smiles)
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
def positional_encoding_config():
    return RelativeDistancePositionalEncodingConfig(
        N_radial_basis_functions=16,
        distance_cutoff=20.0,
        d_projection=64,
        basis_function_config=GaussianBasisConfig(),
    )
