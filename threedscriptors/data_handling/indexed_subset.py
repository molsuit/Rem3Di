from torch.utils.data import Subset
import torch
import numpy as np

class IndexedSubset(Subset):
    def __getattr__(self, name):
        # 1. Try to get the attribute from Subset (e.g. .dataset, .indices)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # 2. Otherwise, fetch it from the base dataset
        attr = getattr(self.dataset, name)

        # 3. Now handle slicing by type
        # 3a. torch.Tensor — use tensor indexing
        if isinstance(attr, torch.Tensor):
            idx_tensor = torch.tensor(self.indices, dtype=torch.long, device=attr.device)
            return attr[idx_tensor]

        # 3b. numpy.ndarray — use ndarray fancy‐indexing
        if isinstance(attr, np.ndarray):
            return attr[np.array(self.indices, dtype=int)]

        # 3c. list or tuple — build a new sequence
        if isinstance(attr, (list, tuple)):
            sliced = [attr[i] for i in self.indices]
            return type(attr)(sliced)

        # 4. Nothing to slice here — return as‐is
        return attr
