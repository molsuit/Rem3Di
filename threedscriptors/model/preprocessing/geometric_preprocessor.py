import torch
from torch import nn

from threedscriptors.configuration.architecture_config import RadialBasisFunctionType
from threedscriptors.data_handling.sample import Sample


class RadialFilter(nn.Module):
    def __init__(self, n_rad, out_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_rad, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, p_geo):  # (N, N, n_rad)
        return self.mlp(p_geo)


class PairDistanceMatrixGeometricPreprocessor(nn.Module):
    def __init__(
        self,
        N_radial_basis_functions: int,
        distance_cutoff: float,
        d_projection: int,
        basis_function_type=RadialBasisFunctionType,
    ):
        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.d_cutoff = distance_cutoff
        self.d_projection = d_projection

        self.radial_basis = basis_function_type.value(
            N_radial_basis_functions, distance_cutoff
        )

        self.proj = nn.Linear(N_radial_basis_functions, d_projection, bias=False)

    def forward(self, sample: Sample):
        positions = sample.atomic_positions.to(dtype=torch.float32)
        atom_mask = sample.padding_mask
        mask_pair = ~(atom_mask[:, :, None] | atom_mask[:, None, :])
        # positions (B, N, 3)
        # Calculate the pairwise distance matrix
        distances = torch.cdist(positions, positions)

        rbf = self.radial_basis(distances)

        P0 = self.proj(rbf)

        P0 = P0 * mask_pair.unsqueeze(-1)
        P0 = 0.5 * (P0 + P0.transpose(1, 2))

        P0 = P0.float()
        rbf = rbf.float()

        return P0, rbf, mask_pair
