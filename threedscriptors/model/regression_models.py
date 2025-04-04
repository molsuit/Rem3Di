from threedscriptors.model.model_builder import build_head
import torch
import torch.nn as nn

from threedscriptors.model.architecture_config import RegressionHeadConfig
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


class MultiTaskRegressionModel(nn.Module):
    def __init__(
        self,
        regression_head_config: list[RegressionHeadConfig],
        encoder: TransformerEncoder,
        task_list: list,
        preprocessor: AtomicDescriptorPreprocess,
        global_aggregator: GlobalAggregator,
    ):
        super().__init__()
        self.encoder = encoder
        input_dim = global_aggregator.config.output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.activation = regression_head_config.activation_fn

        self.task_heads = nn.ModuleDict()
        for head_config in regression_head_config:
            self.task_heads[head_config.name] = build_head(head_config, input_dim)
        self.N_tasks = len(task_list)
        self.preprocessor = preprocessor
        self.global_aggregator = global_aggregator
        self.task_list = task_list

        

    def forward(self, x, padding_mask=None, **kwargs):
        x = self.preprocessor(x)
        x = self.encoder(x, padding_mask)
        x = self.global_aggregator(x)
        x = self.norm(x)
        x = self.activation(x)

        # Compute outputs from each head.
        # Each head's output is assumed to be of shape (batch_size, output_dim)
        preds = []
        for name, head in self.task_heads.items():
            if name in kwargs:
                preds.append(head(x, kwargs[name]))
            else:
                preds.append(head(x))

        # Concatenate outputs along the feature dimension.
        # Final shape: (batch_size, N_tasks * output_dim)
        pred = torch.cat(preds, dim=-1)
        return pred


class FusedMultiHeadRegression(nn.Module):
    # A class that implements the fused calulation of regression heads by using the torch.bmm (batched matrix multiply) instead of sequentially calculating each head.
    pass
