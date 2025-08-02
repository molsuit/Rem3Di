from torch import nn

from threedscriptors.configuration.architecture_config import (
    AttentionAggregatorConfig,
    GlobalAggregatorConfig,
    MeanAggregatorConfig,
)
from threedscriptors.model.pooling import AttnPool, MeanPool


class GlobalAggregator(nn.Module):
    def __init__(self, global_aggregator_config: GlobalAggregatorConfig):
        super().__init__()
        self.config = global_aggregator_config

        print(global_aggregator_config.aggregator_type_config)

        if isinstance(
            global_aggregator_config.aggregator_type_config, AttentionAggregatorConfig
        ):
            self.attn_conf = global_aggregator_config.aggregator_type_config

            self.pool = AttnPool(
                d_in=self.config.input_dim,
                d_hidden=self.attn_conf.head_dim,
                n_heads=self.attn_conf.num_heads,
                dropout=self.attn_conf.attn_dropout,
            )

        elif isinstance(
            global_aggregator_config.aggregator_type_config, MeanAggregatorConfig
        ):

            self.pool = MeanPool()

        else:
            raise ValueError("No pool given")

        if global_aggregator_config.global_molecular_descriptor_dropout is not None:
            self.dropout = nn.Dropout(
                global_aggregator_config.global_molecular_descriptor_dropout
            )

    def forward(self, S, padding_mask):
        out = self.pool(S, padding_mask)

        if self.config.global_molecular_descriptor_dropout is not None:
            out = self.dropout(out)

        return out
