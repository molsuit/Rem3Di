import pydantic_yaml as pyaml
from e3nn.o3 import Irreps
from mace.calculators import MACECalculator

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    RegressionHeadConfig,
)
from threedscriptors.configuration.data_config import DatasetConfig
from threedscriptors.utils.model_utils import (
    get_invariant_indices,
    get_mace_calculator_embedding_dimension,
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
    ):
        self.dataset_config = dataset_config
        self.embedding_preprocessor_config = embedding_preprocessor_config
        self.attention_layer_config = attention_layer_config
        self.encoder_config = encoder_config
        self.global_aggregator_config = global_aggregator_config

        mace_calculator = MACECalculator(self.dataset_config.embedding_model)
        self.initial_irreps = get_mace_calculator_irrep_signature(mace_calculator)
        self.initial_irrep_dim = get_mace_calculator_embedding_dimension(
            mace_calculator
        )

    # A lot of boilerplate that fills in fields in the config

    def process_preprocessor_config(self):
        # Fills in the empty fields of the encoder config from known values
        self.embedding_preprocessor_config.input_irreps = self.initial_irreps
        self.embedding_preprocessor_config.input_embedding_size = self.initial_irrep_dim

        if self.embedding_preprocessor_config.pseudoscalars:
            _, self.embedding_preprocessor_config.output_irreps = get_invariant_indices(
                self.embedding_preprocessor_config.input_irreps + Irreps("128x0o")
            )
        else:
            _, self.embedding_preprocessor_config.output_irreps = get_invariant_indices(
                self.embedding_preprocessor_config.input_irreps
            )

        self.embedding_preprocessor_config.output_irreps_dim = (
            self.embedding_preprocessor_config.output_irreps.dim
        )

    def process_attention_layer_config(self):
        self.attention_layer_config.embedding_dim = (
            self.embedding_preprocessor_config.output_irreps_dim
        )

    def process_global_aggregator_config(self):
        self.global_aggregator_config.input_dim = (
            self.attention_layer_config.embedding_dim
        )

        if isinstance(self.global_aggregator_config.aggregation_fn, list):
            self.global_aggregator_config.output_dim = (
                len(self.global_aggregator_config.aggregation_fn)
                * self.attention_layer_config.embedding_dim
            )
        else:
            self.global_aggregator_config.output_dim = (
                self.global_aggregator_config.input_dim
            )

    def process_regression_heads_config(self) -> list[RegressionHeadConfig]:
        regression_heads = []

        for task in self.dataset_config.tasks:
            if task.auxillary_data_dimension is not None:
                input_dim = (
                    self.global_aggregator_config.output_dim
                    + task.auxillary_data_dimension
                )
            else:
                input_dim = self.global_aggregator_config.output_dim

            regression_heads.append(
                RegressionHeadConfig(
                    task_name=task.task_name, input_dimensions=input_dim
                )
            )

        return regression_heads

    def create_architecture_config_template(self, model_directory):
        # Creates the architecture config with default values and the

        self.process_preprocessor_config()
        self.process_attention_layer_config()
        self.process_global_aggregator_config()
        regression_heads = self.process_regression_heads_config()

        architecture_config = ArchitectureConfig(
            embedding_preprocess_config=self.embedding_preprocessor_config,
            encoder_config=self.encoder_config,
            global_aggregator_config=self.global_aggregator_config,
            regression_head_config=regression_heads,
        )

        pyaml.to_yaml_file(
            f"{model_directory}/architecture_config.yaml", architecture_config
        )

        return architecture_config
