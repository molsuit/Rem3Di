from dataclasses import dataclass, fields

import torch
from torch.utils.data._utils.collate import default_collate


@dataclass
class Sample:
    embeddings: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    regression_targets: torch.Tensor | None = None
    regression_masks: torch.Tensor | None = None
    auxillary_data: dict | None = None
    target_class_labels: torch.Tensor | None = None
    active_decoy_labels: torch.Tensor | None = None
    molecular_descriptors: torch.Tensor | None = None


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
