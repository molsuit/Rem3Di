import math

import torch
import torch.nn as nn
from torch import Tensor

from remedi.configuration.architecture_config import DecoderConfig, ModelDimensions
from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.model.pair_block import PairBlock
from remedi.training.data.sample import PreprocessedSample


class NoiseLevelEmbedding(nn.Module):
    """Embeds a per-molecule noise level: sinusoidal features of its logarithm,
    then a two-layer MLP."""

    def __init__(self, embedding_dim: int, number_of_frequencies: int = 16):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(math.log(0.25), math.log(16.0), number_of_frequencies)
        )
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(2 * number_of_frequencies, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, noise_level: Tensor) -> Tensor:
        """``noise_level`` (B,) > 0 -> (B, embedding_dim)."""
        phases = torch.log(noise_level)[:, None] * self.frequencies
        return self.mlp(torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1))


class TransformerPairDecoder(nn.Module):
    """Noised atomic descriptors + molecular descriptor -> denoising output.

    The noised descriptors are projected to the decoder width and the noise
    level embedding (and, with ``element_tokens``, the element embedding) is
    added to every atom. Each block cross-attends to the descriptor tokens; with
    ``pair_geometry`` the pair representation biases the self-attention; without
    ``atom_self_attention`` the atoms do not attend to each other at all. The
    output head is zero-initialised and maps back to the width of the denoising
    target. Output atom i is compared with target atom i: the order of the
    atoms, not the geometry, pairs predictions with targets.
    """

    #: Embedding rows for atomic numbers 0..118.
    ELEMENT_COUNT = 119

    def __init__(self, decoder_config: DecoderConfig, dimensions: ModelDimensions):
        super().__init__()

        self.config = decoder_config
        attention = decoder_config.attention_layer_config
        embedding_dimension = decoder_config.embedding_dimension

        self.input_projection = nn.Linear(
            dimensions.target_dimension, embedding_dimension
        )
        self.noise_level_embedding = NoiseLevelEmbedding(embedding_dimension)
        self.element_embedding: nn.Embedding | None = None
        if decoder_config.element_tokens:
            self.element_embedding = nn.Embedding(
                self.ELEMENT_COUNT, embedding_dimension
            )
        self.layers = nn.ModuleList(
            [
                PairBlock(
                    embedding_dim=embedding_dimension,
                    num_heads=attention.num_heads,
                    dim_feedforward=attention.dim_feedforward,
                    dropout=attention.dropout,
                    d_pair=dimensions.pair_dimension,
                    d_geo=dimensions.radial_basis_dimension,
                    descriptor_dimension=dimensions.descriptor_dimension,
                    update_pair=(
                        attention.update_pair and index < decoder_config.N_layers - 1
                    ),
                    pair_bias=decoder_config.pair_geometry,
                    self_attention=decoder_config.atom_self_attention,
                )
                for index in range(decoder_config.N_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.output_head = nn.Linear(embedding_dimension, dimensions.target_dimension)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        noised_sample: PreprocessedSample,
        molecular_descriptor: MolecularDescriptor,
        noise_level: Tensor,
    ) -> Tensor:
        """``noised_sample.preprocessed_atomic_embeddings`` (B, N, target) holds
        the (input-scaled) noised descriptors, ``noise_level`` (B,) the noise
        level of each molecule. Returns (B, N, target)."""
        S = self.input_projection(noised_sample.preprocessed_atomic_embeddings)
        S = S + self.noise_level_embedding(noise_level)[:, None, :]
        if self.element_embedding is not None:
            if noised_sample.atomic_numbers is None:
                raise ValueError("element tokens need the padded atomic numbers")
            S = S + self.element_embedding(noised_sample.atomic_numbers.long())
        geometry = self.config.pair_geometry
        P = noised_sample.initial_pair_representation if geometry else None

        for layer in self.layers:
            S, P = layer(
                S,
                noised_sample.padding_mask,
                P,
                noised_sample.geometrical_encoding if geometry else None,
                noised_sample.pair_mask if geometry else None,
                # Cross-attention attends over the full seed sequence (B, L, D).
                descriptor_tokens=molecular_descriptor.tokens,
            )

        return self.output_head(self.final_norm(S))
