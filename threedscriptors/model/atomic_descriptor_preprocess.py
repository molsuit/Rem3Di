import torch
from e3nn import o3
from mace.modules.blocks import tp_out_irreps_with_instructions
from torch import nn

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


class InvariantsFilter(AtomicDescriptorPreprocess):
    def __init__(self, embedding_preprocess_config: EmbeddingPreprocessConfig):
        super().__init__(
            embedding_preprocess_config
        )  # sets the config and indices of invariant reps

        self.config.output_irreps = (
            self.invariant_irreps
        )  # The Irrep string of the invariant concentrators output
        self.config.output_irreps_dim = self.config.output_irreps.dim

    def forward(self, atomic_embedings):
        return remove_equivariants(atomic_embedings, self.invariant_indices)


class PseudoscalarGenerator(AtomicDescriptorPreprocess):
    def __init__(self, embedding_preprocess_config: EmbeddingPreprocessConfig):
        """
        The Pseudoscalar Generator takes in the full atomic descriptor and return the invariant even part of the descriptor concatenated with the new pseudoscalar.
        """
        super().__init__(
            embedding_preprocess_config
        )  # sets the config and indices of invariant reps

        i_in1 = o3.Irreps(self.config.input_irreps)
        embedd_irrep1 = o3.Irreps("10x1o")
        embedd_irrep2 = o3.Irreps("10x1e")

        self.lin0 = o3.Linear(i_in1, embedd_irrep1)
        self.tp_1 = o3.TensorProduct(
            i_in1,
            embedd_irrep1,
            o3.Irreps("128x1e"),
            instructions=[(1, 0, 0, "uvu", True)],
            shared_weights=True,
            internal_weights=True,
        )

        self.lin = o3.Linear(self.tp_1.irreps_out, embedd_irrep2)
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            i_in1,
            self.lin.irreps_out,
            o3.Irreps("1x0o"),
        )
        self.tp_2 = o3.TensorProduct(
            i_in1,
            self.lin.irreps_out,
            irreps_mid,
            instructions=instructions,
            shared_weights=True,
            internal_weights=True,
        )

        self.config.output_irreps = self.invariant_irreps + self.tp_2.irreps_out
        self.config.output_irreps_dim = self.config.output_irreps.dim

    def forward(self, atomic_embedding):
        # calculate the pseudosaclar from in+ equivariant part
        pseudoscalar = self.tp_2(
            atomic_embedding[:],
            self.lin(self.tp_1(atomic_embedding[:], self.lin0(atomic_embedding[:]))),
        )
        # remove the equivariant part
        atomic_embedding = remove_equivariants(atomic_embedding, self.invariant_indices)

        atomic_embedding = torch.cat((atomic_embedding, pseudoscalar), dim=-1)
        return atomic_embedding
