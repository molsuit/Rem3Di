from torch import cat, nn

from threedscriptors.model.architecture_config import GlobalAggregatorConfig


class GlobalAggregator(nn.Module):
    def __init__(self, global_aggregator_config: GlobalAggregatorConfig):
        super().__init__()
        self.config = global_aggregator_config
        self.aggregation_fns = global_aggregator_config.aggregation_fn
        self.config.output_dim = len(self.aggregation_fns) * self.config.input_dim

    def forward(self, x):
        intermediates = [f(x, dim=1) for f in self.aggregation_fns]
        out = cat(intermediates, dim=-1)
        return out
