import torch
import torch.nn as nn

from threedscriptors.model.architecture_config import RegressionHeadConfig
from threedscriptors.model.atomic_descriptor_preprocess import (
    AtomicDescriptorPreprocess,
)
from threedscriptors.model.model import TransformerEncoder


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
        regression_head_config: RegressionHeadConfig,
        encoder: TransformerEncoder,
        task_list: list,
        preprocessor: AtomicDescriptorPreprocess,
    ):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.layers[0].embedding_dim
        self.norm = nn.LayerNorm(input_dim)
        self.activation = regression_head_config.activation_fn

        self.task_heads = nn.ModuleList()
        self.N_tasks = len(task_list)
        regression_head_config.hidden_dimensions.insert(0, input_dim)
        self.preprocessor = preprocessor

        for _ in task_list:
            head = nn.Sequential()

            for idx, dim in enumerate(regression_head_config.hidden_dimensions[:-1]):
                head.add_module(
                    f"linear_{idx}",
                    nn.Linear(dim, regression_head_config.hidden_dimensions[idx + 1]),
                )
                head.add_module("activation", self.activation)

            head.add_module(
                f"linear_{idx + 1}",
                nn.Linear(regression_head_config.hidden_dimensions[-1], 1),
            )

            self.task_heads.append(head)

    def forward(self, x, padding_mask=None):
        x = self.preprocessor(x)
        x = self.encoder(x, padding_mask)
        x = self.norm(x)
        x = self.activation(x)

        # Compute outputs from each head.
        # Each head's output is assumed to be of shape (batch_size, output_dim)
        preds = [head(x) for head in self.task_heads]

        # Concatenate outputs along the feature dimension.
        # Final shape: (batch_size, N_tasks * output_dim)
        pred = torch.cat(preds, dim=-1)
        return pred


class FusedMultiHeadRegression(nn.Module):
    # A class that implements the fused calulation of regression heads by using the torch.bmm (batched matrix multiply) instead of sequentially calculating each head.
    pass
