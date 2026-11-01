import os

# MACE foundation models are pickled modules, which torch >= 2.6 refuses under
# weights_only loading; the MACE tests load them. This conftest sits at the workspace
# root so every test directory (packages/*/tests, tests/integration) gets both settings.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch

# e3nn 0.4.4 loads o3/constants.pt via torch.load; torch>=2.6 defaults
# weights_only=True and rejects the pickled `slice` objects, so `from e3nn import o3`
# (transitively imported by the model code) raises UnpicklingError. Allowlist it.
# Drop this once e3nn is upgraded past the torch.load(weights_only) break.
torch.serialization.add_safe_globals([slice])
