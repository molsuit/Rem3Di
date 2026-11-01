import os

# MACE foundation models are pickled modules, which torch >= 2.6 refuses under
# weights_only loading; the MACE tests load them.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import pytest
import torch

# e3nn 0.4.4 loads o3/constants.pt via torch.load; torch>=2.6 defaults
# weights_only=True and rejects the pickled `slice` objects, so `from e3nn import o3`
# (transitively imported by the model code) raises UnpicklingError. Allowlist it.
# Drop this once e3nn is upgraded past the torch.load(weights_only) break.
torch.serialization.add_safe_globals([slice])


# Used only by test_data_batching.py, whose two tests construct Sample with
# fields it no longer has (cleanup §1.2b).
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
def regression_mask():
    return torch.tensor([1])
