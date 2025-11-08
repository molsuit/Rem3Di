from torch import nn

from threedscriptors.configuration.architecture_config import EncoderConfig
from threedscriptors.data_handling.sample import PreprocessedSample
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.model_output import ModelOutput
from threedscriptors.model.pair_block import PairBlock


class TransformerPairEncoder(nn.Module):
    def __init__(
        self, encoder_config: EncoderConfig, global_aggregator: GlobalAggregator
    ):
        super().__init__()

        self.encoder_config = encoder_config

        self.layers = nn.ModuleList(
            [
                PairBlock(
                    **encoder_config.attention_layer_config.model_dump(),
                    d_pair=self.encoder_config.d_pair,
                    d_geo=self.encoder_config.d_geo,
                )
                for _ in range(encoder_config.N_layers)
            ]
        )

        self.aggregator = global_aggregator

    def forward(self, preprocessed_sample: PreprocessedSample)-> ModelOutput:

        S = preprocessed_sample.preprocessed_atomic_embeddings
        P = preprocessed_sample.initial_pair_representation

        for layer in self.layers:
            S, P = layer(
                S,
                preprocessed_sample.padding_mask,
                P,
                preprocessed_sample.geometrical_encoding,
                preprocessed_sample.pair_mask,
            )

        molecular_descriptor = self.aggregator(S, preprocessed_sample.padding_mask)

        return molecular_descriptor
