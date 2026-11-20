import os

import pytest

# MACE foundation models are pickled modules, which torch >= 2.6 refuses under
# weights_only loading; the MACE tests load them. This conftest sits at the workspace
# root so every test directory (packages/*/tests, tests/integration) gets the setting.
# The e3nn constants (the builtin ``slice``) are allowlisted by ``remedi/__init__.py``.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``gpu``-marked tests on a machine without CUDA. CI deselects them
    (and the ``slow`` ones) with ``-m "not gpu and not slow"``."""
    import torch

    if torch.cuda.is_available():
        return
    skip_without_cuda = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_without_cuda)
