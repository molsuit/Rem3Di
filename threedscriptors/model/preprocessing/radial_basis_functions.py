import torch
import torch.nn as nn
import math

class BesselBasisFunctions(nn.Module):
    """
    DimeNet-style radial Bessel basis functions:
      e_n(d) = sqrt(2 / cutoff) * sin(n * pi * d / cutoff) / d
    for n = 1, ..., N_radial_basis_functions.
    """
    def __init__(self, N_radial_basis_functions: int, distance_cutoff: float, eps: float = 1e-8):
        super().__init__()
        self.N_radial = N_radial_basis_functions
        self.cutoff = distance_cutoff
        self.eps = eps

        # create a buffer of shape [N] with values [1, 2, ..., N]
        n = torch.arange(1, self.N_radial + 1, dtype=torch.float32)
        self.register_buffer("n_idx", n)

        # precompute the overall normalization sqrt(2 / cutoff)
        self.register_buffer("norm", torch.tensor(math.sqrt(2.0 / self.cutoff), dtype=torch.float32))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        distances: Tensor of any shape [...], assumed >= 0
        returns: Tensor of shape [..., N_radial] where
          output[..., i] = norm * sin((i+1)*pi*distances/cutoff) / distances
        """
        # ensure no exact zero for stability
        d = distances.clamp(min=self.eps).unsqueeze(-1)  # shape [..., 1]
        # compute sin(n π d / cutoff) for each n
        arg = self.n_idx * math.pi * d / self.cutoff      # shape [..., N_radial]
        rbf = self.norm * torch.sin(arg) / d               # shape [..., N_radial]
        return rbf


class GaussianBasisFunctions(nn.Module):

    def __init__(self, N_radial_basis_functions: int, distance_cutoff: float):

        super().__init__()

        self.N_radial_basis_functions = N_radial_basis_functions
        self.distance_cutoff = distance_cutoff
        centers = torch.linspace(
            0.0, self.distance_cutoff, self.N_radial_basis_functions
        )
        widths = (
            self.distance_cutoff / self.N_radial_basis_functions
        ) * torch.ones_like(centers)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def forward(self, distances):

        rbf = torch.exp(
            -0.5 * ((distances[..., None] - self.centers) / self.widths) ** 2
        )

        return rbf
