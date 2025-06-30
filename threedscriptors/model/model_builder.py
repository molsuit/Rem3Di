from collections.abc import Sequence

import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
    RandomWalkPositionalEncoding,
    RelativeDistancePositionalEncodingConfig,
)
from threedscriptors.model.atomic_descriptor_preprocess import (
    AtomicDescriptorPreprocess,
    InvariantsFilter,
    PseudoscalarGenerator,
)
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.pair_block import TransformerPairEncoder
from threedscriptors.model.regression_models import (
    MultitaskHeads,
    MultiTaskRegressionModel,
    StructureBasedMultitaskRegressionModel,
)
from threedscriptors.model.structural_encoding import (
    PairDistanceMatrixEncodingBlock,
    RandomWalkStructureEncodingBlock,
)
from threedscriptors.model.transformer_components import TransformerEncoder
from threedscriptors.utils.model_utils import get_invariant_indices


class ModelBuilder:
    def __init__(self, architecture_config: ArchitectureConfig):
        self.architecture_config = architecture_config

        self.model: (
            MultiTaskRegressionModel | StructureBasedMultitaskRegressionModel | None
        ) = None
        self._N_trainable_parameters = None

    @classmethod
    def from_directory(cls, directory: str):
        architecture_config = pyaml.parse_yaml_file_as(
            ArchitectureConfig,
            f"{directory}/architecture_config.yaml",
        )
        return cls(architecture_config)

    @property
    def N_trainable_parameters(self):
        return self._N_trainable_parameters

    @N_trainable_parameters.getter
    def N_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _reload_weights(self):
        if self.architecture_config.reload_full_model_weights:
            self.model.load_state_dict(
                torch.load(self.architecture_config.reload_full_model_weights)
            )
        else:
            if self.architecture_config.embedding_preprocess_config.reload_state_dict:
                self.model.preprocessor.load_state_dict(
                    torch.load(
                        self.architecture_config.embedding_preprocess_config.reload_state_dict
                    )
                )
            if self.architecture_config.encoder_config.reload_state_dict:
                self.model.encoder.load_state_dict(
                    torch.load(
                        self.architecture_config.encoder_config.reload_state_dict
                    )
                )

    def build_model(
        self,
        mean_atomic_embedding=None,
        std_atomic_embedding=None,
        equivariant_scale_factor=None,
    ):
        preprocessor = self.build_preprocess(
            mean_atomic_embedding, std_atomic_embedding, equivariant_scale_factor
        )
        encoder = self.build_encoder()
        aggregator = self.build_global_aggregator()
        multitask_heads = self.build_regression_heads()

        if self.architecture_config.positional_encoding_config is None:
            model = MultiTaskRegressionModel(
                regression_heads=multitask_heads,
                encoder=encoder,
                preprocessor=preprocessor,
                global_aggregator=aggregator,
            )

        elif isinstance(
            self.architecture_config.positional_encoding_config,
            RelativeDistancePositionalEncodingConfig,
        ):

            pos_config = self.architecture_config.positional_encoding_config
            structure_encoding = PairDistanceMatrixEncodingBlock(
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
            structure_encoding = RandomWalkStructureEncodingBlock(
                k_hop=pos_config.k_hop_random_walk, d_projection=pos_config.d_projection
            )

        else:
            raise ValueError("Invalid Choice of Structural Encoding")

        model = StructureBasedMultitaskRegressionModel(
            structure_encoding_block=structure_encoding,
            pair_encoder=encoder,
            preprocessor=preprocessor,
            global_aggregator=aggregator,
            multitask_heads=multitask_heads,
        )

        self.model = model.float()
        self.model.preprocessor.double()

        if (
            self.architecture_config.reload_full_model_weights
            or self.architecture_config.embedding_preprocess_config.reload_state_dict
            or self.architecture_config.encoder_config.reload_state_dict
        ):
            self._reload_weights()

        return model

    def build_preprocess(
        self, mean_atomic_embedding, std_atomic_embedding, equivariant_scale_factor
    ) -> AtomicDescriptorPreprocess:
        preprocess_config = self.architecture_config.embedding_preprocess_config

        if preprocess_config.pseudoscalars:
            preprocessor = PseudoscalarGenerator(preprocess_config)
        else:
            preprocessor = InvariantsFilter(preprocess_config)

        if (mean_atomic_embedding is not None) and (std_atomic_embedding is not None):

            _, invariant_irreps = get_invariant_indices(
                self.architecture_config.embedding_preprocess_config.input_irreps
            )
            invariant_dim = invariant_irreps.dim

            assert mean_atomic_embedding.shape[-1] == invariant_dim
            preprocessor.register_embedding_normalization(
                mean_atomic_embedding, std_atomic_embedding
            )

        if equivariant_scale_factor is not None:

            preprocessor.register_equivariant_scale(equivariant_scale_factor)

        return preprocessor

    def build_encoder(self):
        encoder_config = self.architecture_config.encoder_config

        if encoder_config.d_pair is None:
            encoder = TransformerEncoder(encoder_config)
        else:
            encoder = TransformerPairEncoder(encoder_config=encoder_config)
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
