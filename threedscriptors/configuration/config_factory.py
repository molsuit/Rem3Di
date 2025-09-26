import pydantic_yaml as pyaml

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
    DecoderConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    RandomWalkPositionalEncoding,
    RegressionHeadConfig,
    RelativeDistancePositionalEncodingConfig,
)
from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.utils.model_utils import (
    get_mace_calculator_irrep_signature,
)


class ConfigFactory:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        embedding_preprocessor_config: EmbeddingPreprocessConfig,
        attention_layer_config: AttentionLayerConfig,
        encoder_config: EncoderConfig,
        global_aggregator_config: GlobalAggregatorConfig,
        positional_encoding_config: (
            RelativeDistancePositionalEncodingConfig
            | RandomWalkPositionalEncoding
            | None
        ) = None,
        decoder_config: DecoderConfig | None = None,
    ):
        self.dataset_config = dataset_config
        self.embedding_preprocessor_config = embedding_preprocessor_config
        self.attention_layer_config = attention_layer_config
        self.encoder_config = encoder_config
        self.global_aggregator_config = global_aggregator_config
        self.positional_encoding_config = positional_encoding_config
        self.decoder_config = decoder_config

        mace_calculator = self.dataset_config.embedding_model_config.mace_calc
        self.initial_irreps = get_mace_calculator_irrep_signature(mace_calculator)

    # A lot of boilerplate that fills in fields in the config

    def process_preprocessor_config(self):
        # Fills in the empty fields of the encoder config from known values
        self.embedding_preprocessor_config.input_irreps = self.initial_irreps

    def process_attention_layer_config(self):
        self.attention_layer_config.embedding_dim = (
            self.embedding_preprocessor_config.output_irreps_dim
        )

    def process_global_aggregator_config(self):
        self.global_aggregator_config.input_dim = (
            self.attention_layer_config.embedding_dim
        )

        if self.global_aggregator_config.output_dim is None:
            self.global_aggregator_config.output_dim = (
                self.global_aggregator_config.input_dim
            )

    def process_regression_heads_config(
        self, head_config_template: RegressionHeadConfig
    ) -> list[RegressionHeadConfig]:
        regression_heads = []

        for task in self.dataset_config.tasks:
            head_config = head_config_template.model_copy(deep=True)

            if task.auxillary_data_dimension is not None:
                input_dim = (
                    self.global_aggregator_config.output_dim
                    + task.auxillary_data_dimension
                )
            else:
                input_dim = self.global_aggregator_config.output_dim

            head_config.task_name = task.task_name
            head_config.input_dimensions = input_dim
            regression_heads.append(head_config)

        return regression_heads

    def process_encoder_config(self):
        if self.positional_encoding_config is not None:
            self.encoder_config.d_pair = self.positional_encoding_config.d_projection

            if isinstance(
                self.positional_encoding_config, RandomWalkPositionalEncoding
            ):
                self.encoder_config.d_geo = (
                    self.positional_encoding_config.k_hop_random_walk
                )

            elif isinstance(
                self.positional_encoding_config,
                RelativeDistancePositionalEncodingConfig,
            ):
                self.encoder_config.d_geo = (
                    self.positional_encoding_config.N_radial_basis_functions
                )

    def process_decoder_config(self):
        if self.decoder_config is not None:
            self.decoder_config.d_descriptor = self.global_aggregator_config.output_dim

            self.decoder_config.d_pair = self.positional_encoding_config.d_projection

            if isinstance(
                self.positional_encoding_config, RandomWalkPositionalEncoding
            ):
                self.decoder_config.d_geo = (
                    self.positional_encoding_config.k_hop_random_walk
                )

            elif isinstance(
                self.positional_encoding_config,
                RelativeDistancePositionalEncodingConfig,
            ):
                self.decoder_config.d_geo = (
                    self.positional_encoding_config.N_radial_basis_functions
                )

    def create_architecture_config_template(
        self, model_directory, head_config_template: RegressionHeadConfig
    ):
        # Creates the architecture config with default values and the

        self.process_preprocessor_config()
        self.process_encoder_config()
        self.process_attention_layer_config()
        self.process_global_aggregator_config()
        self.process_decoder_config()

        if self.dataset_config.tasks is not None:
            regression_heads = self.process_regression_heads_config(
                head_config_template
            )

        else:
            regression_heads = None

        architecture_config = ArchitectureConfig(
            embedding_preprocess_config=self.embedding_preprocessor_config,
            encoder_config=self.encoder_config,
            global_aggregator_config=self.global_aggregator_config,
            regression_head_config=regression_heads,
            positional_encoding_config=self.positional_encoding_config,
            decoder_config=self.decoder_config,
        )

        pyaml.to_yaml_file(
            f"{model_directory}/architecture_config.yaml", architecture_config
        )

        return architecture_config
