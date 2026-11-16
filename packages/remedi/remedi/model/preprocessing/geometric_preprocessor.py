import torch
from torch import nn

from remedi.training.data.sample import Sample


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
        radial_basis: nn.Module,
        N_radial_basis_functions: int,
        distance_cutoff: float,
        d_projection: int,
    ):
        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.d_cutoff = distance_cutoff
        self.d_projection = d_projection

        self.radial_basis = radial_basis

        self.proj = nn.Linear(N_radial_basis_functions, d_projection, bias=False)

    def forward(self, sample: Sample):
        positions = sample.atomic_positions.to(dtype=torch.float32)
        atom_mask = sample.padding_mask
        mask_pair = ~(atom_mask[:, :, None] | atom_mask[:, None, :])
        # positions (B, N, 3)
        # Pairwise distance matrix. The default cdist switches to a matmul
        # formulation above 25 atoms, which in float32 loses ~1e-2 Å at
        # realistic coordinates; that made a molecule's descriptor depend on the
        # padded size of its batch. Compute the differences directly instead.
        distances = torch.cdist(
            positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
        )

        rbf = self.radial_basis(distances)

        P0 = self.proj(rbf)

        P0 = P0 * mask_pair.unsqueeze(-1)
        P0 = 0.5 * (P0 + P0.transpose(1, 2))

        P0 = P0.float()
        rbf = rbf.float()

        return P0, rbf, mask_pair
