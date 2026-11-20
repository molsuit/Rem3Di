"""From a flat batch of molecules to the padded layout the featurizer takes.

``MoleculeCollator`` concatenates the molecules' atoms, lists the atom pairs within
the cutoff as flat atom indices, and gives the padding mask of the batch; the mask's
shape carries the longest molecule, so padding needs no host read of the atom counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from remedi.training.data.sample import Sample
from remedi.training.data.samplers import quantize_pad_length

# A padded dimension of size one would be specialised by torch.compile (and recompiled
# for); two atom slots per molecule keep the atom dimension symbolic.
MINIMUM_ATOM_SLOTS = 2


@dataclass
class PaddedMolecules:
    """A batch of molecules in the padded layout ``(molecules, atoms)``."""

    atomic_numbers: torch.Tensor
    """``(molecules, atoms)``, 0 on padding atoms."""
    positions: torch.Tensor
    """``(molecules, atoms, 3)``, each molecule centred on its atoms' mean, 0 on
    padding atoms."""
    padding_mask: torch.Tensor
    """``(molecules, atoms)``, True on padding atoms."""
    edge_index: torch.Tensor
    """``[2, pairs]`` ordered pairs ``(sender, receiver)`` within ``neighbor_cutoff``,
    as indices into the flattened padded atoms: ``molecule * atoms + slot``."""
    neighbor_cutoff: float
    """The cutoff the pairs were selected with."""

    @property
    def atom_counts(self) -> torch.Tensor:
        return (~self.padding_mask).sum(dim=1)


def pad_molecules(sample: Sample, pad_multiple: int = 1) -> PaddedMolecules:
    """The padded layout of a flat batch (``MoleculeCollator``): its atomic numbers
    ``(atoms,)``, positions ``(atoms, 3)``, ``system_index`` ``(atoms,)`` (sorted:
    each molecule's atoms contiguous) and neighbour list (flat atom indices) placed
    by the batch's ``padding_mask`` ``(molecules, longest molecule)``. The atom slots
    are rounded up to a multiple of ``pad_multiple`` (and to at least two). No host
    reads: the consistency checks are device-side asserts.
    """
    if pad_multiple <= 0:
        raise ValueError("pad_multiple must be a positive integer")
    atomic_numbers = sample.atomic_numbers
    positions = sample.atomic_positions
    system_index = sample.system_index
    padding_mask = sample.padding_mask
    edge_index = sample.edge_index
    neighbor_cutoff = sample.neighbor_cutoff
    if (
        atomic_numbers is None
        or positions is None
        or system_index is None
        or padding_mask is None
        or edge_index is None
        or neighbor_cutoff is None
    ):
        raise ValueError(
            "MACE features need a flat batch with atomic positions and numbers, "
            "system_index, the padding mask and the neighbour list (MoleculeCollator)"
        )
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"expected the neighbour list as [2, pairs], got {tuple(edge_index.shape)}"
        )
    molecules, longest = padding_mask.shape
    atom_slots = quantize_pad_length(max(longest, MINIMUM_ATOM_SLOTS), pad_multiple)
    if atom_slots > longest:
        padding_mask = torch.nn.functional.pad(
            padding_mask, (0, atom_slots - longest), value=True
        )
    device = positions.device
    padding_mask = padding_mask.to(device)
    system_index = system_index.to(device=device, dtype=torch.long)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    atom_counts = (~padding_mask).sum(dim=1)
    torch._assert_async(
        atom_counts.sum() == system_index.shape[0],
        "the padding mask and the flat atoms disagree on the atom count",
    )
    torch._assert_async(
        (system_index[1:] >= system_index[:-1]).all(),
        "system_index must be sorted: each molecule's atoms contiguous",
    )
    first_atom = torch.cumsum(atom_counts, dim=0) - atom_counts
    slot = torch.arange(system_index.shape[0], device=device) - first_atom[system_index]
    padded_index = system_index * atom_slots + slot

    padded_atomic_numbers = torch.zeros(
        (molecules, atom_slots), dtype=torch.long, device=device
    )
    padded_atomic_numbers[system_index, slot] = atomic_numbers.to(
        device=device, dtype=torch.long
    )
    padded_positions = positions.new_zeros((molecules, atom_slots, 3))
    padded_positions[system_index, slot] = positions

    real_atom = (~padding_mask)[..., None].to(positions.dtype)
    centre = padded_positions.sum(dim=1, keepdim=True) / atom_counts.clamp_min(1)[
        :, None, None
    ].to(positions.dtype)
    return PaddedMolecules(
        atomic_numbers=padded_atomic_numbers,
        positions=(padded_positions - centre) * real_atom,
        padding_mask=padding_mask,
        edge_index=padded_index[edge_index],
        neighbor_cutoff=float(neighbor_cutoff),
    )
