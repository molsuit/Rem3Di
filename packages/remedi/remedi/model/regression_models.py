from collections import OrderedDict
from itertools import pairwise

import torch
import torch.nn as nn

from remedi.configuration.architecture_config import (
    HeadType,
    RegressionHeadConfig,
    build_activation,
)
from remedi.model.model_output import ModelOutput
from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.model.pair_encoder import TransformerPairEncoder
from remedi.model.preprocessing.preprocessing import Preprocessor
from remedi.training.data.sample import Sample


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation_fn: nn.Module):
        super().__init__()
        # Layer normalization applied to the input.
        self.norm = nn.LayerNorm(out_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation_fn

        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalize the input.
        # Main branch transformation on normalized input.
        out = self.linear(x)
        out = self.activation(out)
        # Residual shortcut
        residual = self.projection(x)
        out = out + residual
        # This activation
        out = self.norm(out)
        # Elementwise addition.
        return out


class FullyConnectedBlock(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, activation_fn: nn.Module, dropout: float
    ):
        super().__init__()

        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("linear_layer", nn.Linear(in_dim, out_dim)),
                    ("layer_norm", nn.LayerNorm(out_dim)),
                    ("activation", activation_fn),
                    ("dropout", nn.Dropout(dropout)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        return out


class RegressionHead(nn.Module):
    def __init__(self, head_config: RegressionHeadConfig, input_dimension: int):
        """Build a regression (or classification) head.

        Args:
            head_config: Configuration for the head.
            input_dimension: Width of the head input (flat descriptor plus the
                task's auxiliary data).
        """

        super().__init__()

        self.head_config = head_config
        # n_classes None -> scalar regression; positive int -> classification
        # logits over n_classes. The whole head body is shared; only the final
        # projection width differs.
        self.n_classes = head_config.n_classes
        self.is_classification = self.n_classes is not None
        out_features = self.n_classes if self.is_classification else 1

        head = nn.Sequential()

        head.add_module("initial_norm", nn.LayerNorm(input_dimension))

        dimensions = [input_dimension, *head_config.hidden_dimensions]
        for index, (in_dim, out_dim) in enumerate(pairwise(dimensions)):
            activation = build_activation(head_config.activation)
            if head_config.head_type == HeadType.FULLY_CONNECTED:
                head.add_module(
                    f"fully_connected_{index}",
                    FullyConnectedBlock(
                        in_dim, out_dim, activation, dropout=head_config.dropout
                    ),
                )
            elif head_config.head_type == HeadType.RESIDUAL:
                head.add_module(
                    f"residual_{index}",
                    ResidualBlock(in_dim, out_dim, activation),
                )
        head.add_module("output_linear", nn.Linear(dimensions[-1], out_features))

        self.head = head

    def forward(self, molecular_descriptor):
        return self.head(molecular_descriptor)

    def inference(self, molecular_descriptor):
        """Regression: the prediction; classification: per-class probabilities."""
        logits_or_value = self.head(molecular_descriptor)
        if self.is_classification:
            return torch.softmax(logits_or_value, dim=-1)
        return logits_or_value


class MultitaskHeads(nn.Module):
    def __init__(
        self,
        regression_head_configs: list[RegressionHeadConfig],
        descriptor_flat_dimension: int,
    ):
        super().__init__()
        self.regression_head_configs = regression_head_configs

        self.task_list = [conf.task_name for conf in regression_head_configs]
        self.N_tasks = len(self.task_list)

        self.task_heads = nn.ModuleDict(
            {
                conf.task_name: RegressionHead(
                    conf,
                    input_dimension=descriptor_flat_dimension
                    + conf.auxiliary_input_dimension,
                )
                for conf in regression_head_configs
            }
        )

    def forward(
        self,
        descriptor: MolecularDescriptor,
        auxillary_data: dict | None = None,
    ):
        flat = descriptor.flat
        preds = []

        for name, head in self.task_heads.items():
            if auxillary_data is not None and name in auxillary_data:
                aux = auxillary_data[name].to(flat.device)
                input_data = torch.cat((flat, aux), dim=1)
                preds.append(head(input_data))
            else:
                preds.append(head(flat))
        # TODO: Make this return a dict of all tasks instead of a stacked tensor to ensure that the task predictions are returned in the correct order. This would require us to also change the way that the dataset yields the regression targets, would also be a dict then. Maybe it should be possible to just assert that the dataset task ordering and the model task ordering are identical.

        preds = torch.cat(preds, dim=-1)

        return preds

    def inference(
        self,
        descriptor: MolecularDescriptor,
        auxillary_data: dict | None = None,
    ):
        flat = descriptor.flat
        preds = []

        for name, head in self.task_heads.items():
            if auxillary_data is not None and name in auxillary_data:
                aux = auxillary_data[name].to(flat.device)
                input_data = torch.cat((flat, aux), dim=1)
                preds.append(head.inference(input_data))
            else:
                preds.append(head.inference(flat))

        preds = torch.cat(preds, dim=-1)

        return preds


class FusedMultiHeadRegression(nn.Module):
    # A class that implements the fused calulation of regression heads by using the torch.bmm (batched matrix multiply) instead of sequentially calculating each head.
    pass


class MultiTaskRegressionModel(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        encoder: TransformerPairEncoder,
        regression_heads: MultitaskHeads,
    ):
        super().__init__()
        self.encoder = encoder
        self.multitask_heads = regression_heads
        self.preprocessor = preprocessor

    def forward(self, sample: Sample) -> ModelOutput:
        molecular_descriptor = self.get_molecular_descriptor(sample)

        preds = self.multitask_heads(molecular_descriptor, sample.auxillary_data)

        return ModelOutput(
            molecular_descriptor=molecular_descriptor, regression_predictions=preds
        )

    def inference(self, sample: Sample) -> ModelOutput:
        # Regression values as predicted, classification as probabilities.

        molecular_descriptor = self.get_molecular_descriptor(sample)

        preds = self.multitask_heads.inference(
            molecular_descriptor, sample.auxillary_data
        )

        return ModelOutput(
            molecular_descriptor=molecular_descriptor, regression_predictions=preds
        )

    def get_molecular_descriptor(self, sample: Sample) -> MolecularDescriptor:
        preprocessed_sample = self.preprocessor(sample)

        molecular_descriptor = self.encoder(preprocessed_sample)
        return molecular_descriptor
