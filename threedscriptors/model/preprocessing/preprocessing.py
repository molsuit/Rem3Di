import torch
from ase.data import atomic_masses
from torch import nn
from torch_sim.models.mace import MaceModel
from torch_sim.state import SimState

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
        if sample.atomic_positions is None or sample.atomic_numbers is None:
            raise ValueError(
                "Sample must provide atomic positions and numbers for torchsim state initialization."
            )

        positions = sample.atomic_positions
        atomic_numbers = sample.atomic_numbers
        system_idx = sample.system_index

        device = positions.device
        dtype = positions.dtype

        atomic_numbers = atomic_numbers.to(device=device, dtype=torch.long)
        system_idx = system_idx.to(device=device, dtype=torch.long)

        # masses[Z] must match atomic_numbers convention (Z or Z-1)
        mass_lookup = _ATOMIC_MASS_TABLE.to(device=device, dtype=dtype)
        masses = mass_lookup.index_select(0, atomic_numbers)

        if system_idx.numel() == 0:
            n_systems = 1
        else:
            if system_idx.min() < 0:
                raise ValueError("system_index must be non-negative.")
            n_systems = int(system_idx.max().item()) + 1

        # large vacuum box, non-periodic
        unit_cell = 500.0 * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        cells = unit_cell.repeat(n_systems, 1, 1)

        torchsim_state = SimState(
            positions=positions,
            masses=masses,
            cell=cells,
            pbc=False,
            atomic_numbers=atomic_numbers,
            system_idx=system_idx,
        )

        with torch.inference_mode():
            out = self.torch_sim_mace_model(torchsim_state)
            atomic_embeddings: torch.Tensor = out["descriptors"].detach()

        # per-system atom counts
        lengths = torch.zeros(n_systems, device=device, dtype=torch.long)
        if system_idx.numel() > 0:
            lengths.index_add_(
                0, system_idx, torch.ones_like(system_idx, dtype=torch.long)
            )

        max_atoms = int(lengths.max().item()) if lengths.numel() > 0 else 0
        atom_dim = positions.shape[-1]
        embed_dim = atomic_embeddings.shape[-1]

        padded_positions = positions.new_zeros((n_systems, max_atoms, atom_dim))
        padded_embeddings = atomic_embeddings.new_zeros(
            (n_systems, max_atoms, embed_dim)
        )

        if system_idx.numel() > 0 and max_atoms > 0:
            sort_idx = torch.argsort(system_idx, stable=True)
            sorted_positions = positions.index_select(0, sort_idx)
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
