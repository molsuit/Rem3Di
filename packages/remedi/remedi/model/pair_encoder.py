from torch import nn

from remedi.configuration.architecture_config import EncoderConfig, ModelDimensions
from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.model.pair_block import PairBlock
from remedi.training.data.sample import PreprocessedSample


class TransformerPairEncoder(nn.Module):
    """Atomic descriptors -> molecular descriptor.

    A learned linear projection takes the (fixed) preprocessed atomic
    descriptors to the model width; the pair blocks follow, then a final layer
    norm and the aggregator.
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        aggregator: nn.Module,
        dimensions: ModelDimensions,
    ):
        super().__init__()

        self.encoder_config = encoder_config
        attention = encoder_config.attention_layer_config

        self.input_projection = nn.Linear(
            dimensions.atomic_descriptor_dimension, dimensions.embedding_dimension
        )
        self.layers = nn.ModuleList(
            [
                PairBlock(
                    embedding_dim=dimensions.embedding_dimension,
                    num_heads=attention.num_heads,
                    dim_feedforward=attention.dim_feedforward,
                    dropout=attention.dropout,
                    d_pair=dimensions.pair_dimension,
                    d_geo=dimensions.radial_basis_dimension,
                    update_pair=(
                        attention.update_pair and index < encoder_config.N_layers - 1
                    ),
                )
                for index in range(encoder_config.N_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dimensions.embedding_dimension)

        self.aggregator = aggregator

    def forward(self, preprocessed_sample: PreprocessedSample) -> MolecularDescriptor:
        S = self.input_projection(preprocessed_sample.preprocessed_atomic_embeddings)
        P = preprocessed_sample.initial_pair_representation

        for layer in self.layers:
            S, P = layer(
                S,
                preprocessed_sample.padding_mask,
                P,
                preprocessed_sample.geometrical_encoding,
                preprocessed_sample.pair_mask,
            )

        tokens = self.aggregator(self.final_norm(S), preprocessed_sample.padding_mask)
        return MolecularDescriptor(tokens=tokens)
