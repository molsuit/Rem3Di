import pydantic_yaml as pyaml

from threedscriptors.configuration.architecture_config import (
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
)
from threedscriptors.configuration.config_factory import ConfigFactory
from threedscriptors.configuration.data_config import DatasetConfig

config_file = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/test/dataset_config.yaml"
dataset_config = pyaml.parse_yaml_file_as(DatasetConfig, config_file)


model_dir = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/test/"


embedding_preprocessor_config = EmbeddingPreprocessConfig(
    pseudoscalars=True, pseudoscalar_dimension=8
)

attention_layer_config = AttentionLayerConfig(
    num_heads=8,
    dim_feedforward=512,
    dropout=0.3,
)

encoder_config = EncoderConfig(
    N_layers=2, attention_layer_config=attention_layer_config
)

global_aggregator_config = GlobalAggregatorConfig(
    aggregation_fn="mean",
)

cf = ConfigFactory(
    dataset_config,
    embedding_preprocessor_config,
    attention_layer_config,
    encoder_config,
    global_aggregator_config,
)

architecture_config = cf.create_architecture_config_template(model_directory=model_dir)
