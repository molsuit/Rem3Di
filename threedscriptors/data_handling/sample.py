from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
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
    embeddings: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    regression_targets: torch.Tensor | None = None
    regression_masks: torch.Tensor | None = None
    auxillary_data: dict[str, Any] | None = None
    atomic_positions: torch.Tensor | None = None
    atomic_numbers: torch.Tensor | None = None
    system_index: torch.Tensor | None = None
    total_charge: torch.Tensor | None = None
    total_spin: torch.Tensor | None = None

    def to(self, device: torch.device, non_blocking: bool = True) -> "Sample":
        moved_fields = {
            name: _move_to(value, device, non_blocking)
            for name, value in self.__dict__.items()
        }
        return Sample(**moved_fields)

    def to_(self, device: torch.device, non_blocking: bool = True) -> "Sample":
        for name, value in list(self.__dict__.items()):
            self.__dict__[name] = _move_to(value, device, non_blocking)
        return self

    def __len__(self):
        return self.atomic_positions.shape[0]

    def pin_memory(self):
        def _pin(x):
            return None if x is None else x.pin_memory()

        return Sample(
            embeddings=_pin(self.embeddings),
            padding_mask=_pin(self.padding_mask),
            regression_targets=_pin(self.regression_targets),
            regression_masks=_pin(self.regression_masks),
            auxillary_data=self.auxillary_data,
            atomic_positions=_pin(self.atomic_positions),
            atomic_numbers=_pin(self.atomic_numbers),
            system_index=_pin(self.system_index),
            total_charge=_pin(self.total_charge),
            total_spin=_pin(self.total_spin),
        )


def pretraining_padded_collate_fn(batch: list[Sample]) -> Sample:
    P_pad = pad_sequence([s.atomic_positions for s in batch], batch_first=True)
    Nmax = P_pad.size(1)
    lengths = torch.tensor([len(s) for s in batch])
    mask = torch.arange(Nmax).expand(len(batch), Nmax) >= lengths.unsqueeze(1)

    if batch[0].embeddings is not None:
        E_pad = pad_sequence([s.embeddings for s in batch], batch_first=True)
        return Sample(embeddings=E_pad, padding_mask=mask, atomic_positions=P_pad)

    return Sample(padding_mask=mask, atomic_positions=P_pad)


def normalization_collate_fn(batch: list[Sample]) -> Sample:
    E = torch.cat(tensors=[s.embeddings for s in batch])
    return Sample(embeddings=E)


def sample_collate_fn(batch: list[Sample]) -> Sample:
    batched = {}
    for f in fields(Sample):
        vals = [getattr(s, f.name) for s in batch]
        if all(v is None for v in vals):
            batched[f.name] = None
        else:
            batched[f.name] = default_collate(vals)
    return Sample(**batched)


def paired_sample_collate_fn(batch: list[tuple["Sample", "Sample"]]):
    left, right = zip(*batch, strict=False)
    batch = sample_collate_fn(list(left) + list(right))
    return batch


def yield_molecules_collate_fn(batch: list[Sample]) -> Sample:
    # Concatenate per-atom tensors and per-system charge/spin scalars from each sample
    # into a single flat batch suitable for on-the-fly MACE embedding.

    atomic_positions = torch.cat([s.atomic_positions for s in batch])
    atomic_numbers = torch.cat([s.atomic_numbers for s in batch])
    repeat_counts = torch.as_tensor(
        [s.atomic_positions.shape[0] for s in batch],
        dtype=torch.long,
        device=atomic_positions.device,
    )
    system_idx = torch.repeat_interleave(
        torch.arange(len(batch), device=atomic_positions.device), repeat_counts
    )

    total_charge = torch.stack(
        [torch.as_tensor(s.total_charge, dtype=atomic_positions.dtype) for s in batch]
    )
    total_spin = torch.stack(
        [torch.as_tensor(s.total_spin, dtype=atomic_positions.dtype) for s in batch]
    )

    return Sample(
        atomic_positions=atomic_positions,
        atomic_numbers=atomic_numbers,
        system_index=system_idx,
        total_charge=total_charge,
        total_spin=total_spin,
    )


@dataclass
class PreprocessedSample:
    preprocessed_atomic_embeddings: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    initial_pair_representation: torch.Tensor | None = None
    geometrical_encoding: torch.Tensor | None = None
    pair_mask: torch.Tensor | None = None
    pair_distance_matrix: torch.Tensor | None = None
    chiral_embeddings: torch.Tensor | None = None
