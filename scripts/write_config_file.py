import os

import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import (
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    HeadType,
    RelativeDistancePositionalEncodingConfig,
    RegressionHeadConfig,
    AttentionAggregatorConfig,
    MeanAggregatorConfig,
    Aggregations
)
from threedscriptors.configuration.config_factory import ConfigFactory
from threedscriptors.configuration.data_config import DatasetConfig

run = "antiviral_potency_only_heavy_atoms"

config_file = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/{run}_full/dataset_config.yaml"
dataset_config = pyaml.parse_yaml_file_as(DatasetConfig, config_file)
 

model_dir = f"/share/snw30/projects/threedscriptor/3DMolecularDescriptors/transformer_model/{run}/"

pos_encoding_config = None

pos_encoding_config = RelativeDistancePositionalEncodingConfig(N_radial_basis_functions=16, distance_cutoff=20, d_projection=64)



embedding_preprocessor_config = EmbeddingPreprocessConfig(
    pseudoscalars=False, pseudoscalar_dimension=0, pseudoscalar_embedding_dim=0,
)

attention_layer_config = AttentionLayerConfig(
    num_heads=8,
    dim_feedforward=1024,
    dropout=0.3,
)

encoder_config = EncoderConfig(
    N_layers=3, attention_layer_config=attention_layer_config
)


mean_aggregator_config = MeanAggregatorConfig(aggregator_type= Aggregations.MEAN)

attention_aggregator_config = AttentionAggregatorConfig(aggregator_type= Aggregations.ATTENTION, num_heads = 16, attn_dropout= 0.3)

global_aggregator_config = GlobalAggregatorConfig(
    aggregator_type_config= attention_aggregator_config,
    global_molecular_descriptor_dropout=0.2
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
