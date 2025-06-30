import torch
from torch import nn

from threedscriptors.model.radial_basis_functions import GaussianBasisFunctions, BesselBasisFunctions
from threedscriptors.configuration.architecture_config import RadialBasisFunctionType


class PairDistanceMatrixEncodingBlock(nn.Module):


    def __init__(self, N_radial_basis_functions: int, distance_cutoff: float, d_projection: int, basis_function_type = RadialBasisFunctionType):

        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.d_cutoff = distance_cutoff
        self.d_projection = d_projection

        
        self.radial_basis = basis_function_type.value(N_radial_basis_functions, distance_cutoff)

        self.proj = nn.Linear(N_radial_basis_functions, d_projection, bias=False)



    def forward(self, positions, atom_mask):

        # positions (B, N, 3)
        # Calculate the pairwise distance matrix
        distances = torch.cdist(positions, positions)


        rbf = self.radial_basis(distances)

        P0  = self.proj(rbf)

        mask_pair = (atom_mask[:, :, None] | atom_mask[:, None, :])

        P0 = P0 * ~mask_pair.unsqueeze(-1)
        P0 = 0.5 * (P0 + P0.transpose(1,2))
        return P0, distances, mask_pair





class RandomWalkStructureEncodingBlock(nn.Module):
    def forward(self, transition_matrix):
        pass
