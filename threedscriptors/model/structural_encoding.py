import torch
from torch import nn
from threedscriptors.data_handling.sample import Sample

from threedscriptors.model.radial_basis_functions import GaussianBasisFunctions, BesselBasisFunctions
from threedscriptors.configuration.architecture_config import RadialBasisFunctionType

from threedscriptors.model.model_output import ModelOutput

class PairDistanceMatrixEncodingBlock(nn.Module):

    def __init__(self, N_radial_basis_functions: int, distance_cutoff: float, d_projection: int, basis_function_type = RadialBasisFunctionType):

        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.d_cutoff = distance_cutoff
        self.d_projection = d_projection

        
        self.radial_basis = basis_function_type.value(N_radial_basis_functions, distance_cutoff)

        self.proj = nn.Linear(N_radial_basis_functions, d_projection, bias=False)


    def forward(self, sample : Sample):
        

        positions = sample.atomic_positions
        atom_mask = sample.padding_mask

        mask_pair = ~(atom_mask[:, :, None] | atom_mask[:, None, :])
        # positions (B, N, 3)
        # Calculate the pairwise distance matrix
        distances = torch.cdist(positions, positions)
        
        rbf = self.radial_basis(distances)
  

        P0  = self.proj(rbf)
        
        P0 = P0 * mask_pair.unsqueeze(-1)
        P0 = 0.5 * (P0 + P0.transpose(1,2))
  


        return ModelOutput(pair_encoding= P0,pair_distances= distances),  mask_pair



class RandomWalkStructureEncodingBlock(nn.Module):
    def __init__(self,k_hop: int, d_projection: int):
        
        super().__init__()

        self.k_hop = k_hop
        self.d_projection = d_projection


        self.proj = nn.Linear(k_hop+1, d_projection, bias=False)

    
    def forward(self, sample: Sample):
        
        transition_matrix = sample.random_walk_transition_matrix
        atom_mask = sample.padding_mask


        mask_pair = ~(atom_mask[:, :, None] | atom_mask[:, None, :])

        powers = []
        B, N, _ = transition_matrix.shape

        # 0-hop = I_N, then zero out padded rows/cols
        I = torch.eye(N, device=transition_matrix.device).unsqueeze(0).expand(B, -1, -1)
        powers.append(I * mask_pair)
        
        # Implement the batch power loop

        T_k = transition_matrix * mask_pair
        powers.append(T_k)

        # 2…k-hop
        for _ in range(2, self.k_hop + 1):
            T_k = torch.matmul(T_k, transition_matrix)
            T_k = T_k * mask_pair
            powers.append(T_k)


        T_stack = torch.stack(powers, dim=-1)
        P0 = self.proj(T_stack)
        return ModelOutput(pair_encoding= P0), mask_pair
