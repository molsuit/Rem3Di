from torch import nn

from threedscriptors.configuration.architecture_config import GlobalAggregatorConfig


class GlobalAggregator(nn.Module):
    def __init__(self, global_aggregator_config: GlobalAggregatorConfig):
        super().__init__()
        self.config = global_aggregator_config

        assert (
            global_aggregator_config.input_dim is not None
            and global_aggregator_config.output_dim is not None
        ), "Dimensions must be resolved by ArchitectureConfig cascade before build."

        self.pool = global_aggregator_config.aggregator_type_config.build(
            input_dim=global_aggregator_config.input_dim,
            output_dim=global_aggregator_config.output_dim,
        )

        if global_aggregator_config.global_molecular_descriptor_dropout is not None:
            self.dropout = nn.Dropout(
                global_aggregator_config.global_molecular_descriptor_dropout
            )

    def forward(self, S, padding_mask):
        out = self.pool(S, padding_mask)

        if self.config.global_molecular_descriptor_dropout is not None:
            out = self.dropout(out)

        return out
