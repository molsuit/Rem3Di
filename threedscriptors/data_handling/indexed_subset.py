from collections.abc import Sequence
from typing import Generic, TypeVar

import torch
from torch.utils.data import Subset

TBase = TypeVar("TBase", bound="BaseDataset")


class IndexedSubset(Subset[TBase], Generic[TBase]):
    dataset: TBase

    def __init__(self, dataset: TBase, indices: Sequence[int]):
        super().__init__(dataset, indices)
        # cache once so we don't re-create tensors every getattr
        self._idx_torch = torch.as_tensor(indices, dtype=torch.long)

    def __getattr__(self, name):
        # 1) try Subset attributes first
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # 2) pull from base dataset
        attr = getattr(self.dataset, name)
        # 3) slice depending on type
        if isinstance(attr, torch.Tensor):
            return attr[self._idx_torch.to(attr.device)]
        if isinstance(attr, (list, tuple)):
            sliced = [attr[i] for i in self.indices]
            return type(attr)(sliced)
        if isinstance(attr, dict):
            # build a new dict, slicing tensor/ndarray/list values
            new = {}
            for k, v in attr.items():
                new[k] = v[self._idx_torch.to(v.device)]
            return new
        # 4) not sliceable → return as-is
        return attr


class IndexedPairedSubset(Subset[TBase], Generic[TBase]):

    def __init__(self, dataset: TBase, indices: Sequence[int]):
        super().__init__(dataset, indices)
        # cache once so we don't re-create tensors every getattr
        self._idx_torch = torch.as_tensor(indices, dtype=torch.long)

        self.structure_indices = (
            torch.Tensor([self.dataset.get_pair_indices(i) for i in self.indices])
            .to(torch.long)
            .reshape(
                -1,
            )
        )

    def __getattr__(self, name):

        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        attr = getattr(self.dataset, name)
        # 3) slice depending on type
        if isinstance(attr, torch.Tensor):
            return attr[self.structure_indices.to(attr.device)]

        if isinstance(attr, dict):
            # build a new dict, slicing tensor/ndarray/list values
            new = {}
            for k, v in attr.items():
                new[k] = v[self.structure_indices.to(attr.device)]

        return attr
