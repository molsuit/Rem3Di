import numpy as np
import torch
from e3nn import o3
from e3nn.nn import BatchNorm
from mace.modules.blocks import tp_out_irreps_with_instructions
from torch import from_numpy, nn
from threedscriptors.model.variance_normalization import VarianceNormalization
from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig
from threedscriptors.utils.model_utils import (
    get_invariant_indices,
    remove_equivariants,
    split_invariants_equivariants,
    get_equivariant_irreps,
    get_pseudoscalar_indices
)


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

        

    def forward(self, atomic_embedding):

        invariants = remove_equivariants(atomic_embedding, self.invariant_indices)

        return self.rescale_invariant(invariants)


class PseudoscalarGenerator(AtomicDescriptorPreprocess):
    def __init__(self, embedding_preprocess_config: EmbeddingPreprocessConfig):

        super().__init__(embedding_preprocess_config)

        self.in_irreps = get_equivariant_irreps(self.config.input_irreps)
        print(self.in_irreps)


        # 1) cross-product: (1o ⊗ 1o) → 1e
        self.tp_cross = o3.TensorProduct(
            self.in_irreps,
            self.in_irreps,
            o3.Irreps("128x1e"),
            # single CG path, weights = CG only
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization = "component",
        )

        self.out_irreps = o3.Irreps(f"128x0o")
        # 2) dot: (1e ⊗ 1o) → 0o
        self.tp_dot = o3.TensorProduct(
            self.tp_cross.irreps_out,
            self.in_irreps,
            self.config.pseudoscalar_irrep,  # final pseudoscalar
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization = "component")

        self.register_buffer(
            "equivariant_scale_factor",
            torch.ones((1, 1, self.config.input_equivariant_dimension)),
            persistent=True,
        )


        self.var_norm = VarianceNormalization(self.config.pseudoscalar_dimension)






    def register_equivariant_scale(self, equivariant_scale_factor):
        
        assert torch.all(self.equivariant_scale_factor == torch.ones((1, 1, self.config.input_equivariant_dimension))), "Equivariant scale buffer has already been set, and can not be overwritten"
        
        self.equivariant_scale_factor = equivariant_scale_factor




    def forward(self, atomic_embeddings):


        invariant_features, equivariant_features = split_invariants_equivariants(atomic_embeddings, self.invariant_indices)

        invariant_features = self.rescale_invariant(invariant_features)    
        
        equivariant_features = equivariant_features * self.equivariant_scale_factor

        cross = self.tp_cross(equivariant_features, equivariant_features)  # v₂ × v₃

        chi = self.tp_dot(equivariant_features, cross)  # v₁ · (v₂ × v₃)
        
        #chi = self.var_norm(chi)
        chi = chi.float()



        atomic_descriptors = torch.cat((invariant_features, chi), dim = -1)


        return atomic_descriptors



    def slice_pseudoscalars(self, processed_atomic_descriptors):
        
        pseudo_slices = get_pseudoscalar_indices(self.config.output_irreps)
        blocks = [processed_atomic_descriptors[..., slc] for slc in pseudo_slices]

        return torch.cat(blocks, dim=-1) 
    
    


