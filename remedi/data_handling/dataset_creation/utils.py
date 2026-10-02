import numpy as np
import torch
from torch import Tensor


def system_idx_to_ragged_ptr(system_idx: Tensor) -> Tensor:
    """
    Convert a per-atom `system_idx` vector into a ragged pointer (cumulative end
    indices), optionally forcing the pointer length to be `n_mols`.

    Using `minlength=n_mols` ensures we include trailing zero-count systems when
    the last system(s) contributed no atoms, keeping lengths consistent with
    metadata like `structure_ids`.
    """

    counts = torch.bincount(system_idx)
    ptr = torch.cumsum(counts, dim=0)
    return ptr


def ensure_numpy_array(array: Tensor | np.ndarray | None):
    if array is None:
        return None

    array = (
        array.detach().cpu().numpy()
        if isinstance(array, torch.Tensor)
        else np.asarray(array)
    )

    return array
