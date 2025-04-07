import torch
import torch.nn as nn

from threedscriptors.configuration.architecture_config import RegressionHeadConfig
from threedscriptors.model.atomic_descriptor_preprocess import (
    AtomicDescriptorPreprocess,
)
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.transformer_components import TransformerEncoder


class SingleRegressionModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, encoder: TransformerEncoder):
        super().__init__()

        self.encoder = encoder
        input_dim = encoder.layers[0].embedding_dim

        self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.SiLU()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x, padding_mask=None):
        x = self.encoder(x, padding_mask)
        x = self.norm(x)
        x = self.activation(x)
        x = self.linear1(x)
        return x


class FusedMultiHeadRegression(nn.Module):
    # A class that implements the fused calulation of regression heads by using the torch.bmm (batched matrix multiply) instead of sequentially calculating each head.
    pass


class MultitaskHeads(nn.Module):
    def __init__(self, regression_head_configs: list[RegressionHeadConfig]):
        super().__init__()
        self.regression_head_configs = regression_head_configs

        self.task_list = [conf.task_name for conf in regression_head_configs]
        self.N_tasks = len(self.task_list)

        self.task_heads = nn.ModuleDict()
        for head_config in regression_head_configs:
            self.task_heads[head_config.task_name] = self.build_head(head_config)

    @staticmethod
    def build_head(
        head_config: RegressionHeadConfig,
    ) -> nn.Module:
        """
        Build a regression head based on the provided configuration.

        Args:
            head_config (RegressionHeadConfig): Configuration for the regression head.
            input_dim (int): Input dimension for the regression head.

        Returns:
            nn.Module: The constructed regression head.
        """
        head = nn.Sequential()

        # TODO: Should there be an initial layer normalization, and activation function immediately after the molecular descriptor?

        for idx, dim in enumerate(head_config.hidden_dimensions[:-1]):
            head.add_module(
                f"linear_{idx}",
                nn.Linear(dim, head_config.hidden_dimensions[idx + 1]),
            )
            head.add_module("activation", head_config.activation_fn)

        head.add_module(
            f"linear_{idx + 1}",
            nn.Linear(head_config.hidden_dimensions[-1], 1),
        )

        return head

    def forward(self, descriptor, auxillary_data: dict | None = None):
        preds = []
        for name, head in self.task_heads.items():
            if auxillary_data is not None and name in auxillary_data:
                input_data = torch.cat((descriptor, auxillary_data["name"]))
                preds.append(head(input_data))
            else:
                preds.append(head(descriptor))


class MultiTaskRegressionModel(nn.Module):
    def __init__(
        self,
        regression_heads: MultitaskHeads,
        encoder: TransformerEncoder,
        preprocessor: AtomicDescriptorPreprocess,
        global_aggregator: GlobalAggregator,
    ):
        super().__init__()
        self.encoder = encoder
        self.multitask_heads = regression_heads
        self.preprocessor = preprocessor
        self.global_aggregator = global_aggregator

    def forward(self, x, padding_mask=None, auxillary_data: dict | None = None):
        x = self.preprocessor(x)
        x = self.encoder(x, padding_mask)
        descriptor = self.global_aggregator(x)

        preds = self.multitask_heads(descriptor, auxillary_data)
        # Compute outputs from each head.
        # Each head's output is assumed to be of shape (batch_size, output_dim)

        # Concatenate outputs along the feature dimension.
        # Final shape: (batch_size, N_tasks * output_dim)
        pred = torch.cat(preds, dim=-1)
        return pred
