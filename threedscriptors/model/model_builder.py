from collections.abc import Sequence

import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import ArchitectureConfig
from threedscriptors.model.preprocessing.preprocessing import (
    Preprocessor,
    PreprocessorWithAtomicEmbedding,
)
from threedscriptors.model.regression_models import (
    MultitaskHeads,
    MultiTaskRegressionModel,
)
from threedscriptors.model.remedi_model import REM3DIModel


class ModelBuilder:
    def __init__(self, architecture_config: ArchitectureConfig):
        self.architecture_config = architecture_config
        self.model: MultiTaskRegressionModel | None = None

    @classmethod
    def from_directory(cls, directory: str, trained: bool = True):
        filename = (
            "post_training_architecture_config.yaml"
            if trained
            else "architecture_config.yaml"
        )
        architecture_config = pyaml.parse_yaml_file_as(
            ArchitectureConfig, f"{directory}/{filename}"
        )
        return cls(architecture_config)

    @property
    def N_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def insert_task_configs_into_regression_heads(self, task_configs):
        for task_cfg, head_cfg in zip(
            task_configs, self.architecture_config.regression_head_config, strict=False
        ):
            assert head_cfg.task_name == task_cfg.task_name
            head_cfg.task_config = task_cfg

    def _reload_model_weights(self):
        self.model.load_state_dict(
            torch.load(self.architecture_config.reload_full_model_weights)
        )

    def build_preprocessor(
        self, mean_atomic_embedding=None, std_atomic_embedding=None
    ) -> Preprocessor:
        return Preprocessor(
            atomic_preprocessor=self.architecture_config.embedding_preprocess_config.build(
                mean_atomic_embedding, std_atomic_embedding
            ),
            geometric_preprocessor=self.architecture_config.positional_encoding_config.build(),
        )

    def build_preprocessor_with_mace_embedding(
        self, mace_model, mean_atomic_embedding=None, std_atomic_embedding=None
    ) -> PreprocessorWithAtomicEmbedding:
        return PreprocessorWithAtomicEmbedding(
            mace_model=mace_model,
            atomic_preprocessor=self.architecture_config.embedding_preprocess_config.build(
                mean_atomic_embedding, std_atomic_embedding
            ),
            geometric_preprocessor=self.architecture_config.positional_encoding_config.build(),
        )

    def build_encoder(self):
        global_aggregator = self.architecture_config.global_aggregator_config.build()
        return self.architecture_config.encoder_config.build(global_aggregator)

    def build_decoder(self):
        return self.architecture_config.decoder_config.build()

    def build_regression_heads(self) -> MultitaskHeads:
        head_configs = self.architecture_config.regression_head_config
        if isinstance(head_configs, Sequence):
            return MultitaskHeads(regression_head_configs=head_configs)
        raise NotImplementedError

    def build_model(self, mean_atomic_embedding=None, std_atomic_embedding=None):
        preprocessor = self.build_preprocessor(
            mean_atomic_embedding, std_atomic_embedding
        )
        encoder = self.build_encoder()
        multitask_heads = self.build_regression_heads()

        model = MultiTaskRegressionModel(
            preprocessor=preprocessor,
            encoder=encoder,
            regression_heads=multitask_heads,
        )

        self.model = model.float()
        self.model.preprocessor.atomic_preprocessor.double()

        if self.architecture_config.reload_full_model_weights:
            self._reload_model_weights()

        return self.model

    def build_remedi_model(self, mace_calc=None) -> REM3DIModel:
        preprocessor = self.build_preprocessor()
        encoder = self.build_encoder()

        model = REM3DIModel(
            preprocessor=preprocessor, encoder=encoder, mace_calculator=mace_calc
        )
        self.model = model.float()
        self.model.preprocessor.atomic_preprocessor.double()

        if self.architecture_config.reload_full_model_weights:
            self._reload_model_weights()

        return self.model
