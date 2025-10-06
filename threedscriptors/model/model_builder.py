from collections.abc import Sequence

import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
    RandomWalkPositionalEncoding,
    RelativeDistancePositionalEncodingConfig,
)
from threedscriptors.model.decoder import TransformerDecoder, TransformerPairDecoder
from threedscriptors.model.encoder import TransformerEncoder
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.pair_encoder import TransformerPairEncoder
from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocessor,
)
from threedscriptors.model.preprocessing.geometric_preprocessor import (
    PairDistanceMatrixGeometricPreprocessor,
    RandomWalkGeometricPreprocessor,
)
from threedscriptors.model.preprocessing.preprocessing import Preprocessor
from threedscriptors.model.regression_models import (
    MultitaskHeads,
    MultiTaskRegressionModel,
)
from threedscriptors.model.remedi_model import REM3DIModel
from threedscriptors.utils.model_utils import get_invariant_indices


class ModelBuilder:
    def __init__(self, architecture_config: ArchitectureConfig):
        self.architecture_config = architecture_config

        self.model: MultiTaskRegressionModel | None = None

        self._N_trainable_parameters = None

    @classmethod
    def from_directory(cls, directory: str, trained: bool = True):
        if trained:
            path = f"{directory}/post_training_architecture_config.yaml"
        else:
            path = f"{directory}/architecture_config.yaml"
        architecture_config = pyaml.parse_yaml_file_as(
            ArchitectureConfig,
            path,
        )
        return cls(architecture_config)

    @property
    def N_trainable_parameters(self):
        return self._N_trainable_parameters

    @N_trainable_parameters.getter
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


    def build_remedi_model(self, mace_calc = None) -> REM3DIModel:

        preprocessor = self.build_preprocessor(None,None)
        encoder = self.build_encoder()

        model = REM3DIModel(preprocessor=preprocessor, encoder=encoder, mace_calculator= mace_calc)
        self.model = model.float()
        self.model.preprocessor.atomic_preprocessor.double()

        if self.architecture_config.reload_full_model_weights:
            self._reload_model_weights()

        return self.model

    def build_geometric_preprocessing(self):

        if isinstance(
            self.architecture_config.positional_encoding_config,
            RelativeDistancePositionalEncodingConfig,
        ):
            pos_config = self.architecture_config.positional_encoding_config
            structure_encoding = PairDistanceMatrixGeometricPreprocessor(
                N_radial_basis_functions=pos_config.N_radial_basis_functions,
                distance_cutoff=pos_config.distance_cutoff,
                d_projection=pos_config.d_projection,
                basis_function_type=pos_config.basis_function_type,
            )

        elif isinstance(
            self.architecture_config.positional_encoding_config,
            RandomWalkPositionalEncoding,
        ):

            pos_config = self.architecture_config.positional_encoding_config
            structure_encoding = RandomWalkGeometricPreprocessor(
                k_hop=pos_config.k_hop_random_walk, d_projection=pos_config.d_projection
            )

        else:
            raise ValueError("Invalid Choice of Structural Encoding")

        if pos_config.reload_state_dict is not None:

            structure_encoding.load_state_dict(torch.load(pos_config.reload_state_dict))

        return structure_encoding

    def build_atomic_preprocessor(
        self,embedding_preprocess_config, mean_atomic_embedding= None, std_atomic_embedding = None
    ,) -> AtomicDescriptorPreprocessor:
        preprocess_config = embedding_preprocess_config

        atomic_preprocessor = AtomicDescriptorPreprocessor(preprocess_config=preprocess_config)


        if (mean_atomic_embedding is not None) and (std_atomic_embedding is not None):

            _, invariant_irreps = get_invariant_indices(
                embedding_preprocess_config.input_irreps
            )
            invariant_dim = invariant_irreps.dim

            assert mean_atomic_embedding.shape[-1] == invariant_dim

            atomic_preprocessor.invariant_normalization.set_stats(mean = mean_atomic_embedding, std= std_atomic_embedding)


        if preprocess_config.reload_state_dict is not None:
            print(preprocess_config.reload_state_dict)
            atomic_preprocessor.load_state_dict(
                torch.load(preprocess_config.reload_state_dict)
            )


        return atomic_preprocessor

    def build_preprocessor(
        self, mean_atomic_embedding, std_atomic_embedding
    ) -> Preprocessor:

        atomic_preprocessor = self.build_atomic_preprocessor(
            self.architecture_config.embedding_preprocess_config, mean_atomic_embedding, std_atomic_embedding
        )

        if self.architecture_config.positional_encoding_config is not None:

            geometric_preprocessor = self.build_geometric_preprocessing()

            return Preprocessor(
                atomic_preprocessor=atomic_preprocessor,
                geometric_preprocessor=geometric_preprocessor,
            )

        return Preprocessor(atomic_preprocessor=atomic_preprocessor, geometric_preprocessor=None)

    def build_decoder(self) -> TransformerDecoder:
        return TransformerPairDecoder(self.architecture_config.decoder_config)

    def build_encoder(self):
        encoder_config = self.architecture_config.encoder_config
        global_aggregator = self.build_global_aggregator()

        if encoder_config.d_pair is None:
            encoder = TransformerEncoder(encoder_config, global_aggregator)
        else:
            encoder = TransformerPairEncoder(
                encoder_config=encoder_config, global_aggregator=global_aggregator
            )

        if encoder_config.reload_state_dict:

            encoder.load_state_dict(torch.load(encoder_config.reload_state_dict),strict=False)

        return encoder

    def build_global_aggregator(self) -> GlobalAggregator:
        return GlobalAggregator(self.architecture_config.global_aggregator_config)

    def build_regression_heads(self) -> MultitaskHeads:
        regression_head_config = self.architecture_config.regression_head_config

        if isinstance(regression_head_config, Sequence):
            regression_heads = MultitaskHeads(
                regression_head_configs=regression_head_config
            )

            return regression_heads
        else:
            # Build single regression head, but should probably get rid of this as the single regression head could also be multihead with tasks  = [task]
            raise NotImplementedError
