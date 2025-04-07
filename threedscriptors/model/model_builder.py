from collections.abc import Sequence

import torch.nn as nn

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
)
from threedscriptors.model.atomic_descriptor_preprocess import (
    AtomicDescriptorPreprocess,
    InvariantsFilter,
    PseudoscalarGenerator,
)
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.regression_models import (
    MultitaskHeads,
    MultiTaskRegressionModel,
)
from threedscriptors.model.transformer_components import TransformerEncoder


class ModelBuilder:
    def __init__(self, architecture_config: ArchitectureConfig):
        self.architecture_config = architecture_config
        self.model: nn.Module | None = None
        self._N_trainable_parameters = None

    @property
    def N_trainable_parameters(self):
        return self._N_trainable_parameters

    @N_trainable_parameters.getter
    def N_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def build_model(self):
        preprocessor = self.build_preprocess()
        encoder = self.build_encoder()
        aggregator = self.build_global_aggregator()
        multitask_heads = self.build_regression_heads()

        model = MultiTaskRegressionModel(
            regression_heads=multitask_heads,
            encoder=encoder,
            preprocessor=preprocessor,
            global_aggregator=aggregator,
        )

        return model

    def build_preprocess(self) -> AtomicDescriptorPreprocess:
        preprocess_config = self.architecture_config.embedding_preprocess_config

        if preprocess_config.pseudoscalars:
            preprocessor = PseudoscalarGenerator(preprocess_config)
        else:
            preprocessor = InvariantsFilter(preprocess_config)

        return preprocessor

    def build_encoder(self):
        encoder_config = self.architecture_config.encoder_config
        encoder = TransformerEncoder(encoder_config)
        return encoder

    def build_global_aggregator(self):
        global_aggregator_config = self.architecture_config.global_aggregator_config

        global_aggregator = GlobalAggregator(global_aggregator_config)

        return global_aggregator

    def build_regression_heads(self):
        regression_head_config = self.architecture_config.regression_head_config

        if isinstance(regression_head_config, Sequence):
            regression_heads = MultitaskHeads(
                regression_head_configs=regression_head_config
            )

            return regression_heads
        else:
            # Build single regression head
            raise NotImplementedError
