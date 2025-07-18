from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate


def _move_to(x: Any, device: torch.device, non_blocking: bool) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=non_blocking)
    elif isinstance(x, Mapping):
        return {k: _move_to(v, device, non_blocking) for k, v in x.items()}
    elif isinstance(x, Sequence) and not isinstance(x, str | bytes):
        return type(x)(_move_to(v, device, non_blocking) for v in x)
    else:
        return x  # includes None, scalars, objects


@dataclass
class Sample:
    embeddings:            torch.Tensor | None = None
    padding_mask:          torch.Tensor | None = None
    regression_targets:    torch.Tensor | None = None
    regression_masks:      torch.Tensor | None = None
    auxillary_data:        dict[str, Any] | None = None
    target_class_labels:   torch.Tensor | None = None
    active_decoy_labels:   torch.Tensor | None = None
    molecular_descriptors: torch.Tensor | None = None
    atomic_positions:      torch.Tensor | None = None
    random_walk_transition_matrix: torch.Tensor | None = None

    def to(self, device: torch.device, non_blocking: bool = True) -> "Sample":
        moved_fields = {
            name: _move_to(value, device, non_blocking)
            for name, value in self.__dict__.items()
        }
        return Sample(**moved_fields)

    def to_(self, device: torch.device, non_blocking: bool = True) -> "Sample":
        # The inplace version of moving a Sample to a device

        for name, value in list(self.__dict__.items()):
            self.__dict__[name] = _move_to(value, device, non_blocking)
        return self


def sample_collate_fn(batch: list[Sample]) -> Sample:
    batched = {}
    for f in fields(Sample):
        vals = [getattr(s, f.name) for s in batch]
        # if every sample has None, we keep None
        if all(v is None for v in vals):
            batched[f.name] = None
        else:
            # for auxillary_data this will batch each dict key automatically,
            # and for tensors it will stack them
            batched[f.name] = default_collate(vals)
    return Sample(**batched)



@dataclass
class PreprocessedSample:
    preprocessed_atomic_embeddings : torch.Tensor | None = None
    padding_mask:          torch.Tensor | None = None
    initial_pair_representation: torch.Tensor | None = None
    geometrical_encoding: torch.Tensor | None = None
    pair_mask: torch.Tensor | None = None
    pair_distance_matrix: torch.Tensor | None = None