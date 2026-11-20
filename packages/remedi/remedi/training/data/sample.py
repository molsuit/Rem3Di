import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from remedi.training.data.neighbor_list import molecule_neighbor_list


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
    auxiliary_data: dict[str, Any] | None = None
    atomic_positions: torch.Tensor | None = None
    atomic_numbers: torch.Tensor | None = None
    system_index: torch.Tensor | None = None
    total_charge: torch.Tensor | None = None
    multiplicity: torch.Tensor | None = None
    # The neighbour list (MoleculeCollator): ordered atom pairs within
    # ``neighbor_cutoff``, as flat atom indices [2, pairs].
    edge_index: torch.Tensor | None = None
    neighbor_cutoff: float | None = None

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

    def pin_memory(self) -> "Sample":
        return Sample(
            **{
                name: value.pin_memory() if torch.is_tensor(value) else value
                for name, value in self.__dict__.items()
            }
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


@dataclass(frozen=True)
class MoleculeCollator:
    """Collates molecules into one flat batch for on-the-fly MACE features.

    The batch holds the per-atom positions, atomic numbers and molecule index
    (``system_index``), the per-molecule charge and multiplicity, the padding mask
    ``(molecules, longest molecule)`` (True on padding) of the padded layout the
    features are computed in (its shape gives the padded length without a host
    read on the device), and the neighbour list: the ordered atom pairs of a
    molecule closer than ``cutoff``, as flat atom indices ``[2, pairs]``.

    ``cutoff`` must be the MACE model's ``r_max`` (``MaceConfig.r_max`` or
    ``MaceFeaturizer.r_max``): the batch records it as ``neighbor_cutoff`` and the
    featurizer rejects a batch built for another cutoff. ``with_regression_targets``
    also stacks each molecule's scalar ``regression_targets`` into a ``(molecules,)``
    tensor (supervised training from raw atoms).

    A plain dataclass of plain values, so DataLoader workers can pickle it.
    """

    cutoff: float
    with_regression_targets: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.cutoff) or self.cutoff <= 0.0:
            raise ValueError(
                f"the neighbour cutoff must be positive, got {self.cutoff}"
            )

    def __call__(self, batch: list[Sample]) -> Sample:
        if not batch:
            raise ValueError("cannot collate an empty batch of molecules")
        molecule_positions: list[torch.Tensor] = []
        molecule_atomic_numbers: list[torch.Tensor] = []
        total_charges: list[torch.Tensor] = []
        multiplicities: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for sample in batch:
            if (
                sample.atomic_positions is None
                or sample.atomic_numbers is None
                or sample.total_charge is None
                or sample.multiplicity is None
            ):
                raise ValueError(
                    "every molecule needs atomic positions and numbers, its total "
                    "charge and its multiplicity"
                )
            dtype = sample.atomic_positions.dtype
            molecule_positions.append(sample.atomic_positions)
            molecule_atomic_numbers.append(sample.atomic_numbers)
            total_charges.append(torch.as_tensor(sample.total_charge, dtype=dtype))
            multiplicities.append(torch.as_tensor(sample.multiplicity, dtype=dtype))
            if self.with_regression_targets:
                if sample.regression_targets is None:
                    raise ValueError("every molecule needs a regression target")
                targets.append(sample.regression_targets.reshape(()))
        atomic_positions = torch.cat(molecule_positions)
        atom_counts = torch.as_tensor(
            [positions.shape[0] for positions in molecule_positions], dtype=torch.long
        )
        system_index = torch.repeat_interleave(torch.arange(len(batch)), atom_counts)
        longest = int(atom_counts.max())
        padding_mask = torch.arange(longest)[None, :] >= atom_counts[:, None]

        return Sample(
            atomic_positions=atomic_positions,
            atomic_numbers=torch.cat(molecule_atomic_numbers),
            system_index=system_index,
            padding_mask=padding_mask,
            total_charge=torch.stack(total_charges).to(atomic_positions.dtype),
            multiplicity=torch.stack(multiplicities).to(atomic_positions.dtype),
            regression_targets=torch.stack(targets) if targets else None,
            edge_index=molecule_neighbor_list(molecule_positions, self.cutoff),
            neighbor_cutoff=self.cutoff,
        )


@dataclass
class PreprocessedSample:
    # What the encoder reads: normalised invariants (+ chiral embedding).
    preprocessed_atomic_embeddings: torch.Tensor | None = None
    # The fixed (parameter-free) normalised invariants: the denoising target.
    normalized_invariants: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    initial_pair_representation: torch.Tensor | None = None
    geometrical_encoding: torch.Tensor | None = None
    pair_mask: torch.Tensor | None = None
    pair_distance_matrix: torch.Tensor | None = None
    chiral_embeddings: torch.Tensor | None = None
    # (B, N) atomic numbers, 0 at padding.
    atomic_numbers: torch.Tensor | None = None
