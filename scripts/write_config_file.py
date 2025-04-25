import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import (
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    HeadType,
    RegressionHeadConfig,
)
from threedscriptors.configuration.config_factory import ConfigFactory
from threedscriptors.configuration.data_config import DatasetConfig

config_file = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/data/tmqm/dataset_config.yaml"
dataset_config = pyaml.parse_yaml_file_as(DatasetConfig, config_file)


model_dir = "/data/fast-pc-06/snw30/projects/threescriptor/3DMolecularDescriptors/transformer_model/tmqm/"


embedding_preprocessor_config = EmbeddingPreprocessConfig(
    pseudoscalars=True, pseudoscalar_dimension=128, pseudoscalar_embedding_dim=128
)

attention_layer_config = AttentionLayerConfig(
    num_heads=8,
    dim_feedforward=256,
    dropout=0.3,
)

encoder_config = EncoderConfig(
    N_layers=1, attention_layer_config=attention_layer_config
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

head_config_template = RegressionHeadConfig(
    activation_fn=torch.nn.SiLU(),
    hidden_dimensions=[256, 128],
    head_type=HeadType.RESIDUAL,
)


architecture_config = cf.create_architecture_config_template(
    model_directory=model_dir, head_config_template=head_config_template
)
