import torch
import torch.nn as nn

from threedprints.model.model import TransformerEncoder


class SingleRegressionModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, encoder: TransformerEncoder):
        super().__init__()

        self.encoder = encoder
        input_dim = encoder.layers[0].embedding_dim

        self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.SiLU()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, padding_mask=None):
        x = self.encoder(x, padding_mask)
        # x = self.norm(x)
        x = self.activation(x)
        x = self.linear1(x)
        # x= self.norm2(x)
        # x = self.activation(x)
        # x = self.linear2(x)

        return x


class MultiTaskRegressionModel(nn.Module):
    def __init__(
        self, hidden_dim, output_dim, encoder: TransformerEncoder, task_list: list
    ):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.layers[0].embedding_dim
        self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.SiLU()

        self.task_heads = nn.ModuleList()
        self.N_tasks = len(task_list)

        for _ in task_list:
            # For regression, a simple linear layer can be sufficient.
            head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, output_dim),
            )
            self.task_heads.append(head)

    def forward(self, x, padding_mask=None):
        # Pass through the encoder and activation modules.
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
