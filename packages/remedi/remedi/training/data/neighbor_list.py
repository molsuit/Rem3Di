"""The neighbour list of a batch of molecules: the atom pairs within a cutoff.

Built on the CPU when a batch is collated (positions are never noised, so the list
is fixed per batch): the MACE featurizer then runs on these pairs alone instead of
every pair within a molecule, and the pair count reaches it as an input shape rather
than as a data-dependent size inside its graph.
"""

from __future__ import annotations

import math

import torch
from torch.nn.utils.rnn import pad_sequence


def molecule_neighbor_list(
    molecule_positions: list[torch.Tensor], cutoff: float
) -> torch.Tensor:
    """Every ordered pair ``(sender, receiver)`` of distinct atoms of one molecule
    that are closer than ``cutoff``, as ``[2, pairs]`` int64 indices into the
    concatenation of the molecules' atoms (the flat layout of a collated batch).

    ``molecule_positions``: one ``(atoms, 3)`` tensor per molecule. Pairs are
    ordered by molecule, then sender, then receiver. Distances are computed in
    float64, whatever the positions' dtype. A molecule without pairs (one atom, or
    atoms farther apart than ``cutoff``) contributes none.
    """
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError(f"the neighbour cutoff must be positive, got {cutoff}")
    if not molecule_positions:
        return torch.zeros((2, 0), dtype=torch.long)
    for positions in molecule_positions:
        if positions.dim() != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError(
                f"expected positions (atoms, 3) with atoms > 0, got "
                f"{tuple(positions.shape)}"
            )
    atom_counts = torch.tensor(
        [positions.shape[0] for positions in molecule_positions], dtype=torch.long
    )
    padded_positions = pad_sequence(
        [positions.to(torch.float64) for positions in molecule_positions],
        batch_first=True,
    )
    atom_slots = padded_positions.shape[1]
    slot = torch.arange(atom_slots)
    real_atom = slot[None, :] < atom_counts[:, None]
    distances = torch.cdist(
        padded_positions,
        padded_positions,
        compute_mode="donot_use_mm_for_euclid_dist",
    )
    within_cutoff = (
        (distances < cutoff)
        & real_atom[:, :, None]
        & real_atom[:, None, :]
        & (slot[:, None] != slot[None, :])
    )
    molecule, sender_slot, receiver_slot = within_cutoff.nonzero(as_tuple=True)
    first_atom = torch.cumsum(atom_counts, dim=0) - atom_counts
    return torch.stack(
        [first_atom[molecule] + sender_slot, first_atom[molecule] + receiver_slot]
    )
