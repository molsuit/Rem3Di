from collections.abc import Sequence

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
        self.model: MultiTaskRegressionModel | None = None
        self._N_trainable_parameters = None

    @property
    def N_trainable_parameters(self):
        return self._N_trainable_parameters

    @N_trainable_parameters.getter
    def N_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _reload_weights(self):
        if self.architecture_config.reload_full_model_weights:
            self.model.load_state_dict(
                self.architecture_config.reload_full_model_weights
            )
        else:
            if self.architecture_config.embedding_preprocess_config.reload_state_dict:
                self.model.preprocessor.load_state_dict(
                    self.architecture_config.embedding_preprocess_config.reload_state_dict
                )
            if self.architecture_config.encoder_config.reload_state_dict:
                self.model.encoder.load_state_dict(
                    self.architecture_config.encoder_config.reload_state_dict
                )

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
        self.model = model

        if (
            self.architecture_config.reload_full_model_weights
            or self.architecture_config.embedding_preprocess_config.reload_state_dict
            or self.architecture_config.encoder_config.reload_state_dict
        ):
            self._reload_weights()

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
            # Build single regression head, but should probably get rid of this as the single regression head could also be multihead with tasks  = [task]
            raise NotImplementedError
