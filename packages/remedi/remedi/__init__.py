"""Rem3Di: smooth, chiral 3D molecular descriptors from atomistic foundation models."""

import torch

# e3nn 0.4.4 (the version mace-torch pins) reads its Wigner constants with a plain
# ``torch.load`` when it is imported. Since torch 2.6 that load is weights-only and
# refuses the builtin ``slice`` the file holds, so every module that imports e3nn
# failed without TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD. Allowlisting ``slice`` makes
# them importable; loading a MACE model file still needs the variable.
torch.serialization.add_safe_globals([slice])
