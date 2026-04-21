import torch
from ase.data import atomic_masses
from torch import nn
from torch_sim.models.mace import MaceModel
from torch_sim.state import SimState
from torch_sim.typing import SystemExtras

from threedscriptors.data_handling.sample import PreprocessedSample, Sample
from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocessor,
)
from threedscriptors.model.preprocessing.geometric_preprocessor import (
    PairDistanceMatrixGeometricPreprocessor,
)

_ATOMIC_MASS_TABLE = torch.as_tensor(atomic_masses, dtype=torch.float32)


class Preprocessor(nn.Module):
    def __init__(
        self,
        atomic_preprocessor,
        geometric_preprocessor: PairDistanceMatrixGeometricPreprocessor,
    ):
        super().__init__()

        self.atomic_preprocessor: AtomicDescriptorPreprocessor = atomic_preprocessor
        self.geometric_preprocessor = geometric_preprocessor

    def forward(self, sample: Sample) -> PreprocessedSample:
        preprocessed_sample: PreprocessedSample = self.atomic_preprocessor(
            sample.embeddings, sample.padding_mask
        )

        preprocessed_sample.padding_mask = sample.padding_mask

        (
            preprocessed_sample.initial_pair_representation,
            preprocessed_sample.geometrical_encoding,
            preprocessed_sample.pair_mask,
        ) = self.geometric_preprocessor(sample)

        return preprocessed_sample


def _sample_to_simstate(sample: Sample, r_max: float) -> tuple[SimState, torch.Tensor]:
    """Build a torch-sim SimState directly from flat Sample tensors.

    Mirrors the cell/position prep that the md_clustering pipeline does in ASE
    (`_prepare_atoms_for_md`): per-system centering, extent sized for torch-sim's
    fixed neighbour list (padding = max(20 Å, 5·r_max)), and shift into
    [0, extent). Vectorised across the batch so we never hit a Python for-loop
    over systems.

    Returns the SimState together with the per-system atom counts tensor so the
    caller can redistribute outputs into padded per-system tensors.
    """
    if sample.atomic_positions is None or sample.atomic_numbers is None:
        raise ValueError(
            "Sample must provide atomic positions and numbers for torchsim state initialization."
        )
    if sample.system_index is None:
        raise ValueError("Sample must provide system_index for batched inference.")

    positions = sample.atomic_positions
    atomic_numbers = sample.atomic_numbers.to(device=positions.device, dtype=torch.long)
    system_idx = sample.system_index.to(device=positions.device, dtype=torch.long)

    device = positions.device
    dtype = positions.dtype

    if system_idx.numel() == 0:
        raise ValueError("Cannot build SimState from an empty batch.")
    if system_idx.min() < 0:
        raise ValueError("system_index must be non-negative.")
    n_systems = int(system_idx.max().item()) + 1

    lengths = torch.bincount(system_idx, minlength=n_systems)

    sums = torch.zeros((n_systems, 3), device=device, dtype=dtype)
    sums.index_add_(0, system_idx, positions)
    means = sums / lengths.to(dtype).unsqueeze(1).clamp_min(1.0)
    centered = positions - means.index_select(0, system_idx)

    per_atom_norm = centered.norm(dim=-1)
    radius = torch.zeros(n_systems, device=device, dtype=dtype).scatter_reduce(
        0, system_idx, per_atom_norm, reduce="amax", include_self=False
    )

    padding = max(20.0, 5.0 * float(r_max))
    extent = 2.0 * (radius + padding)  # (n_systems,)

    shifted_positions = centered + (extent / 2.0).index_select(0, system_idx).unsqueeze(
        -1
    )

    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    cells = eye * extent.view(n_systems, 1, 1)

    mass_lookup = _ATOMIC_MASS_TABLE.to(device=device, dtype=dtype)
    masses = mass_lookup.index_select(0, atomic_numbers)

    system_extras: dict[str, torch.Tensor] = {}
    if sample.total_charge is not None:
        system_extras[SystemExtras.TOTAL_CHARGE] = sample.total_charge.to(
            device=device, dtype=dtype
        ).reshape(n_systems)
    if sample.total_spin is not None:
        system_extras[SystemExtras.TOTAL_SPIN] = sample.total_spin.to(
            device=device, dtype=dtype
        ).reshape(n_systems)

    state = SimState(
        positions=shifted_positions,
        masses=masses,
        cell=cells,
        pbc=False,
        atomic_numbers=atomic_numbers.to(torch.int),
        system_idx=system_idx,
        _system_extras=system_extras,
    )
    return state, lengths


class PreprocessorWithAtomicEmbedding(Preprocessor):
    def __init__(
        self,
        mace_model: MaceModel,
        atomic_preprocessor,
        geometric_preprocessor: PairDistanceMatrixGeometricPreprocessor,
    ):
        super().__init__(atomic_preprocessor, geometric_preprocessor)
        self.torch_sim_mace_model = mace_model

    def forward(self, sample: Sample) -> PreprocessedSample:
        state, lengths = _sample_to_simstate(
            sample, r_max=float(self.torch_sim_mace_model.r_max)
        )

        with torch.inference_mode():
            out = self.torch_sim_mace_model(state)
            atomic_embeddings: torch.Tensor = out["node_feats"].detach()

        n_systems = lengths.shape[0]
        max_atoms = int(lengths.max().item()) if lengths.numel() > 0 else 0

        device = state.positions.device
        atom_dim = state.positions.shape[-1]
        embed_dim = atomic_embeddings.shape[-1]

        padded_positions = sample.atomic_positions.new_zeros(
            (n_systems, max_atoms, atom_dim)
        )
        padded_embeddings = atomic_embeddings.new_zeros(
            (n_systems, max_atoms, embed_dim)
        )

        # Positions come from the (already-centred) per-system view so that the
        # downstream pair-distance preprocessor sees the same coordinates
        # the MACE model saw.
        sort_idx = torch.argsort(state.system_idx, stable=True)
        sorted_positions = sample.atomic_positions.index_select(0, sort_idx)
        sorted_embeddings = atomic_embeddings.index_select(0, sort_idx)

        start = 0
        for sys_id, count in enumerate(lengths.tolist()):
            if count == 0:
                continue
            end = start + count
            padded_positions[sys_id, :count] = sorted_positions[start:end]
            padded_embeddings[sys_id, :count] = sorted_embeddings[start:end]
            start = end

        if max_atoms > 0:
            padding_mask = torch.arange(max_atoms, device=device).unsqueeze(0).expand(
                n_systems, max_atoms
            ) >= lengths.unsqueeze(1)
        else:
            padding_mask = torch.zeros((n_systems, 0), device=device, dtype=torch.bool)

        sample.embeddings = padded_embeddings
        sample.atomic_positions = padded_positions
        sample.padding_mask = padding_mask

        return super().forward(sample)
