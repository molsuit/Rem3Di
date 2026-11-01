from collections import OrderedDict
from itertools import pairwise

import torch
import torch.nn as nn

from remedi.configuration.architecture_config import (
    HeadType,
    RegressionHeadConfig,
)
from remedi.configuration.data_config import LabelScalingType
from remedi.data_handling.sample import Sample
from remedi.model.model_output import ModelOutput
from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.model.pair_encoder import TransformerPairEncoder
from remedi.model.preprocessing.preprocessing import Preprocessor


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
    def __init__(self, in_dim: int, out_dim: int, activation_fn: nn.Module):
        super().__init__()

        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("linear_layer", nn.Linear(in_dim, out_dim)),
                    ("layer_norm", nn.LayerNorm(out_dim)),
                    ("activation", nn.SiLU()),
                    ("dropout", nn.Dropout(0.1)),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        out = self.block(x)
        return out


class RegressionHead(nn.Module):
    def __init__(self, head_config: RegressionHeadConfig):
        """
        Build a regression head based on the provided configuration.

        Args:
            head_config (RegressionHeadConfig): Configuration for the regression head.
            input_dim (int): Input dimension for the regression head.

        Returns:
            nn.Module: The constructed regression head.
        """

        super().__init__()

        self.head_config = head_config
        self.task_config = head_config.task_config
        # n_classes None -> scalar regression; positive int -> classification
        # logits over n_classes. The whole head body is shared; only the final
        # projection width and the (regression-only) label scaling differ.
        self.n_classes = head_config.n_classes
        self.is_classification = self.n_classes is not None
        out_features = self.n_classes if self.is_classification else 1

        head = nn.Sequential()

        head.add_module("initial_norm", nn.LayerNorm(head_config.input_dimensions))

        dimensions = [head_config.input_dimensions, *head_config.hidden_dimensions]
        idx = 0
        for idx, (in_dim, out_dim) in enumerate(pairwise(dimensions)):
            if head_config.head_type == HeadType.FULLY_CONNECTED:
                head.add_module(
                    f"fully_connected_{idx}",
                    FullyConnectedBlock(in_dim, out_dim, head_config.activation_fn),
                )
            elif head_config.head_type == HeadType.RESIDUAL:
                head.add_module(
                    f"residual_{idx}",
                    ResidualBlock(in_dim, out_dim, head_config.activation_fn),
                )
        head.add_module(
            f"linear_{idx + 1}",
            nn.Linear(dimensions[-1], out_features),
        )

        self.head = head

        # Classification heads emit raw logits — no label standardization, and
        # task_config (with its mean/std/scaling) is not required.
        if not self.is_classification:
            # Default to identity scaling when no task_config was inserted
            # (e.g. a bare build() before insert_task_configs).
            mean = getattr(self.task_config, "mean", 0.0) if self.task_config else 0.0
            std = getattr(self.task_config, "std", 1.0) if self.task_config else 1.0
            if type(mean) is float:
                mean = torch.Tensor([mean])
            if type(std) is float:
                std = torch.Tensor([std])

            self.register_buffer("task_mean", mean)
            self.register_buffer("task_std", std)

    def forward(self, molecular_descriptor):
        return self.head(molecular_descriptor)

    def inference(self, molecular_descriptor):
        logits_or_value = self.head(molecular_descriptor)

        # Classification: return per-class probabilities (softmax over logits).
        if self.is_classification:
            return torch.softmax(logits_or_value, dim=-1)

        # Regression: undo the label standardization.
        standardized_prediction = (logits_or_value * self.task_std) + self.task_mean

        if getattr(self.task_config, "scaling", None) == LabelScalingType.LOG_Z:
            standardized_prediction = torch.exp(standardized_prediction)

        return standardized_prediction


class MultitaskHeads(nn.Module):
    def __init__(self, regression_head_configs: list[RegressionHeadConfig]):
        super().__init__()
        self.regression_head_configs = regression_head_configs

        self.task_list = [conf.task_name for conf in regression_head_configs]
        self.N_tasks = len(self.task_list)

        self.task_heads = nn.ModuleDict(
            {conf.task_name: RegressionHead(conf) for conf in regression_head_configs}
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

    def forward(self, sample: Sample):
        molecular_descriptor = self.get_molecular_descriptor(sample)

        preds = self.multitask_heads(molecular_descriptor, sample.auxillary_data)

        return ModelOutput(
            molecular_descriptor=molecular_descriptor, regression_predictions=preds
        )

    def inference(self, sample: Sample):
        # Calls the multitask inference method that returns the prediction in the original unit system

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
