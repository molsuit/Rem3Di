import numpy as np
import torch
from e3nn import o3
from torch import from_numpy, nn

from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig

from threedscriptors.utils.model_utils import (
    get_equivariant_irreps,
    get_invariant_indices,
    get_pseudoscalar_indices,
    remove_equivariants,
    split_invariants_equivariants,
)


class RMSLayerNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        RMS-style layer norm for equivariant (type-L) blocks.

        Args:
          num_channels: number of irreducible blocks C
          eps: small constant to avoid div/0
        """
        super().__init__()
        # one learnable scale per channel/block
        self.gamma = nn.Parameter(torch.ones(num_channels, 1))
        self.eps = eps

    def forward(self, S_e: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: tensor of shape (..., C, D)

        Returns:
          normalized tensor of the same shape
        """
        # 1) compute per-block L2 norm over the last dim:
        #    shape: (..., C, 1)

        B, N, D = S_e.shape
        C = D // 3

        S_e = S_e.view(B, N, C, 3)

        block_norms = torch.linalg.norm(S_e, dim=-1, keepdim=True)

        # 2) compute RMS of those norms *across* the C channels:
        #    shape: (..., 1, 1)
        rms = torch.sqrt(
            torch.mean(block_norms.pow(2), dim=-2, keepdim=True) + self.eps
        )

        if padding_mask is not None:
            rms = rms.masked_fill(padding_mask[..., None, None], 1.0)

        # 3) divide each block by the shared rms and apply per-channel scale
        #    broadcasting gamma over any leading dims and over D
        out = (S_e / rms) * self.gamma

        if padding_mask is not None:
            out = out.masked_fill(padding_mask[..., None, None], 0.0)

        return out.reshape(B, N, D)


class AtomicDescriptorPreprocess(nn.Module):
    """
    Pretreats the calculated embeddings with two possible strategies. 1. Get only the invariant part. 2. Add the pseudoscalar.
    """

    def __init__(self, preprocess_config: EmbeddingPreprocessConfig):
        super().__init__()
        self.config = preprocess_config
        self.invariant_indices, self.invariant_irreps = get_invariant_indices(
            self.config.input_irreps
        )

        self.register_buffer(
            "mean_inv_atomic_embedding",
            torch.zeros((1, 1, self.config.input_invariant_dimension)),
            persistent=True,
        )
        self.register_buffer(
            "std_inv_atomic_embedding",
            torch.ones((1, 1, self.config.input_invariant_dimension)),
            persistent=True,
        )

    def register_embedding_normalization(
        self, mean_atomic_embedding, std_atomic_embedding
    ):
        # Should add a buffer that contains the mean and std deviation of the descriptor, which can be enable before loading.
        assert torch.all(
            self.mean_inv_atomic_embedding
            == torch.zeros_like(self.mean_inv_atomic_embedding)
        ), "Mean atomic embedding buffer has already been set, and can not be overwritten"

        assert torch.all(
            self.std_inv_atomic_embedding
            == torch.ones_like(self.std_inv_atomic_embedding)
        ), "Std atomic embedding buffer has already been set, and can not be overwritten"

        if isinstance(mean_atomic_embedding, np.ndarray):
            mean_atomic_embedding = from_numpy(mean_atomic_embedding)
        if isinstance(std_atomic_embedding, np.ndarray):
            std_atomic_embedding = from_numpy(std_atomic_embedding)

        self.mean_inv_atomic_embedding = mean_atomic_embedding

        self.std_inv_atomic_embedding = std_atomic_embedding

    def rescale_invariant(self, invariants):

        invariants = (
            invariants - self.mean_inv_atomic_embedding
        ) / self.std_inv_atomic_embedding

        invariants = invariants.float()
        return invariants


class InvariantsFilter(AtomicDescriptorPreprocess):
    def __init__(self, embedding_preprocess_config: EmbeddingPreprocessConfig):
        super().__init__(
            embedding_preprocess_config
        )  # sets the config and indices of invariant reps

    def forward(self, atomic_embedding, padding_mask):

        invariants = remove_equivariants(atomic_embedding, self.invariant_indices)

        invariants = self.rescale_invariant(invariants)
        invariants = invariants * (~padding_mask[..., None]).float()
        return invariants


class PseudoscalarGenerator(AtomicDescriptorPreprocess):
    def __init__(self, embedding_preprocess_config: EmbeddingPreprocessConfig):

        super().__init__(embedding_preprocess_config)

        self.in_irreps = get_equivariant_irreps(self.config.input_irreps)

        # 1) cross-product: (1o ⊗ 1o) → 1e
        self.tp_cross = o3.TensorProduct(
            self.in_irreps,
            self.in_irreps,
            o3.Irreps("128x1e"),
            # single CG path, weights = CG only
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        self.out_irreps = o3.Irreps("128x0o")
        # 2) dot: (1e ⊗ 1o) → 0o
        self.tp_dot = o3.TensorProduct(
            self.tp_cross.irreps_out,
            self.in_irreps,
            self.config.pseudoscalar_irrep,  # final pseudoscalar
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        self.rms_norm = RMSLayerNorm(self.in_irreps.num_irreps)
        self.ln = nn.LayerNorm(self.config.output_irreps_dim, dtype=torch.float32)



    def forward(self, atomic_embeddings, padding_mask):

        invariant_features, equivariant_features = split_invariants_equivariants(
            atomic_embeddings, self.invariant_indices
        )

        invariant_features = self.rescale_invariant(invariant_features)

        equivariant_features = self.rms_norm(equivariant_features, padding_mask)
        #

        cross = self.tp_cross(equivariant_features, equivariant_features)  # v₂ x v₃
        chi = self.tp_dot(equivariant_features, cross)  # v₁ · (v₂ x v₃)

        if padding_mask is not None:
            chi = chi * (~padding_mask[..., None]).float()

        chi = chi.to(torch.float32)

        atomic_descriptors = torch.cat(
            (invariant_features.float(), chi.float()), dim=-1
        )

        # self.ln.to(torch.float32)

        # atomic_descriptors = self.ln(atomic_descriptors)

        return atomic_descriptors

    def slice_pseudoscalars(self, processed_atomic_descriptors):

        pseudo_slices = get_pseudoscalar_indices(self.config.output_irreps)
        blocks = [processed_atomic_descriptors[..., slc] for slc in pseudo_slices]

        return torch.cat(blocks, dim=-1)
