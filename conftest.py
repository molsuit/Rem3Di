import os

# MACE foundation models are pickled modules, which torch >= 2.6 refuses under
# weights_only loading; the MACE tests load them. This conftest sits at the workspace
# root so every test directory (packages/*/tests, tests/integration) gets the setting.
# The e3nn constants (the builtin ``slice``) are allowlisted by ``remedi/__init__.py``.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
