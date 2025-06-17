import numpy as np
import torch
from e3nn import o3
from e3nn.nn import BatchNorm
from mace.modules.blocks import tp_out_irreps_with_instructions
from torch import from_numpy, nn

from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig
from threedscriptors.utils.model_utils import get_invariant_indices, remove_equivariants


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
        print(self.config.input_embedding_size)
        self.register_buffer(
            "mean_atomic_embedding",
            torch.zeros((1, 1, self.config.input_embedding_size)),
            persistent=True,
        )
        self.register_buffer(
            "std_atomic_embedding",
            torch.ones((1, 1, self.config.input_embedding_size)),
            persistent=True,
        )

    def register_embedding_normalization(
        self, mean_atomic_embedding, std_atomic_embedding
    ):
        # Should add a buffer that contains the mean and std deviation of the descriptor, which can be enable before loading.
        assert torch.all(
            self.mean_atomic_embedding == torch.zeros_like(self.mean_atomic_embedding)
        ), "Mean atomic embedding buffer has already been set, and can not be overwritten"

        assert torch.all(
            self.std_atomic_embedding == torch.ones_like(self.std_atomic_embedding)
        ), "Std atomic embedding buffer has already been set, and can not be overwritten"

        if isinstance(mean_atomic_embedding, np.ndarray):
            mean_atomic_embedding = from_numpy(mean_atomic_embedding)
        if isinstance(std_atomic_embedding, np.ndarray):
            std_atomic_embedding = from_numpy(std_atomic_embedding)

        self.mean_atomic_embedding = mean_atomic_embedding

        self.std_atomic_embedding = std_atomic_embedding


class InvariantsFilter(AtomicDescriptorPreprocess):
    def __init__(self, embedding_preprocess_config: EmbeddingPreprocessConfig):
        super().__init__(
            embedding_preprocess_config
        )  # sets the config and indices of invariant reps

        self.config.output_irreps = (
            self.invariant_irreps
        )  # The Irrep string of the invariant concentrators output
        self.config.output_irreps_dim = self.config.output_irreps.dim

    def forward(self, atomic_embedding):
        atomic_embedding = (
            atomic_embedding - self.mean_atomic_embedding
        ) / self.std_atomic_embedding

        return remove_equivariants(atomic_embedding, self.invariant_indices)


class PseudoscalarGenerator(AtomicDescriptorPreprocess):
    def __init__(self, embedding_preprocess_config: EmbeddingPreprocessConfig):
        """
        The Pseudoscalar Generator takes in the full atomic descriptor and return the invariant even part of the descriptor concatenated with the new pseudoscalar.
        """
        super().__init__(
            embedding_preprocess_config
        )  # sets the config and indices of invariant reps

        i_in1 = o3.Irreps(self.config.input_irreps)
        embedd_irrep1 = o3.Irreps(
            f"{embedding_preprocess_config.pseudoscalar_embedding_dim}x1o"
        )
        embedd_irrep2 = o3.Irreps(
            f"{embedding_preprocess_config.pseudoscalar_embedding_dim}x1e"
        )

        out_irrep = o3.Irreps(
            f"{embedding_preprocess_config.pseudoscalar_dimension}x0o"
        )

        self.lin0 = o3.Linear(i_in1, embedd_irrep1, internal_weights = True)
        self.tp_1 = o3.TensorProduct(
            i_in1,
            embedd_irrep1,
            o3.Irreps("128x1e"),
            instructions=[(1, 0, 0, "uvu", True)],
            shared_weights=True,
            internal_weights=True,
        )

        self.lin = o3.Linear(self.tp_1.irreps_out, embedd_irrep2, internal_weights = True)
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            i_in1,
            self.lin.irreps_out,
            o3.Irreps("1x0o"),
        )
        print(self.lin.irreps_out)
        print(irreps_mid)
        print(instructions)
        
        

        self.tp_2 = o3.TensorProduct(
            i_in1,
            self.lin.irreps_out,
            irreps_mid,
            instructions=instructions,
            shared_weights=True,
            internal_weights=True,
 #           normalization="component",          # <- per‑component std 1
 #           path_normalization="element", 
        )
        self.out_lin = o3.Linear(self.tp_2.irreps_out, out_irrep)
        print(self.out_lin.irreps_out)
        print(self.tp_2.irreps_out)

        self.norm = torch.nn.LayerNorm(embedding_preprocess_config.pseudoscalar_dimension)

        self.config.output_irreps = self.invariant_irreps + self.out_lin.irreps_out

        self.config.output_irreps_dim = self.config.output_irreps.dim

    def forward(self, atomic_embedding):
        # apply the atomic embedding normalization
        # Not allowed!!!!!!! Cannot normalize equivariant features like this

        raise ValueError()
        atomic_embedding = (
            atomic_embedding - self.mean_atomic_embedding
        ) / self.std_atomic_embedding

        # calculate the pseudosaclar from in+ equivariant part
        pseudoscalar = self.get_pseudoscalars(atomic_embedding)

        # remove the equivariant part
        atomic_embedding = remove_equivariants(atomic_embedding, self.invariant_indices)

        atomic_embedding = torch.cat((atomic_embedding, pseudoscalar), dim=-1)
        return atomic_embedding

    def get_pseudoscalars(self, atomic_embedding):
        pseudoscalar = self.out_lin(self.tp_2(
            atomic_embedding[:],
            self.lin(self.tp_1(atomic_embedding[:], self.lin0(atomic_embedding[:]))),
        ))

        pseudoscalar = self.norm(pseudoscalar)
    

        return pseudoscalar
