import pydantic_yaml as pyaml
import torch
import os
from threedscriptors.configuration.architecture_config import (
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    HeadType,
    RegressionHeadConfig,
    PositionalEncodingConfig
)
from threedscriptors.configuration.config_factory import ConfigFactory
from threedscriptors.configuration.data_config import DatasetConfig

run = "cmrt"

config_file = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{run}_full/dataset_config.yaml"
dataset_config = pyaml.parse_yaml_file_as(DatasetConfig, config_file)


model_dir = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/{run}/"

#pos_encoding_config = None

pos_encoding_config = PositionalEncodingConfig(N_radial_basis_functions=16, distance_cutoff=32, d_projection=64)



embedding_preprocessor_config = EmbeddingPreprocessConfig(
    pseudoscalars=True, pseudoscalar_dimension=128, pseudoscalar_embedding_dim=10,
)

attention_layer_config = AttentionLayerConfig(
    num_heads=8,
    dim_feedforward=1024,
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
    positional_encoding_config=pos_encoding_config
)

head_config_template = RegressionHeadConfig(
    activation_fn=torch.nn.SiLU(),
    hidden_dimensions=[256,128],
    head_type=HeadType.FULLY_CONNECTED,
)


os.makedirs(model_dir, exist_ok= True)

architecture_config = cf.create_architecture_config_template(
    model_directory=model_dir, head_config_template=head_config_template
)
