import torch
from torch import nn


class BesselBasisFunctions(nn.Module):

    # DimeNet Style Bessel Basis functions
    def __init__(self, N_radial_basis_functions: int, distance_cutoff: float):

        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.distance_cutoff = distance_cutoff


    def forward(distances):
        pass



class GaussianBasisFunctions(nn.Module):

    def __init__(self, N_radial_basis_functions: int, distance_cutoff: float):

        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.distance_cutoff = distance_cutoff


        centers = torch.linspace(0., self.distance_cutoff, self.N_radial_basis_functions)

        widths = (self.distance_cutoff / self.N_radial_basis_functions) * torch.ones_like(centers)

        self.register_buffer('centers', centers)
        self.register_buffer('widths',  widths)



    def forward(self, distances):

        rbf = torch.exp(-0.5 * ((distances[..., None] - self.centers) / self.widths)**2)

        return rbf


class PairDistanceMatrixEncodingBlock(nn.Module):


    def __init__(self, N_radial_basis_functions: int, distance_cutoff: float, d_projection: int):

        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.d_cutoff = distance_cutoff
        self.d_projection = d_projection

        self.radial_basis = GaussianBasisFunctions(N_radial_basis_functions, distance_cutoff)

        self.proj = nn.Linear(N_radial_basis_functions, d_projection, bias=False)



    def forward(self, positions, atom_mask):

        # positions (B, N, 3)
        # Calculate the pairwise distance matrix
        distances = torch.cdist(positions, positions)

        # TODO: Bessel functions???


        rbf = self.radial_basis(distances)

        P0  = self.proj(rbf)

        mask_pair = (atom_mask[:, :, None] | atom_mask[:, None, :])

        P0 = P0 * ~mask_pair.unsqueeze(-1)
        P0 = 0.5 * (P0 + P0.transpose(1,2))
        return P0, distances, mask_pair





class RandomWalkStructureEncodingBlock(nn.Module):
    def forward(self, degree_matrix, adjacency_matrix):
        pass
