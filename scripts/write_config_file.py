import os

import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import (
    Aggregations,
    AttentionAggregatorConfig,
    AttentionLayerConfig,
    DecoderConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    HeadType,
    MeanAggregatorConfig,
    RadialBasisFunctionType,
    RegressionHeadConfig,
    RelativeDistancePositionalEncodingConfig,
)
from threedscriptors.configuration.config_factory import ConfigFactory
from threedscriptors.configuration.data_config import DatasetConfig

run = "tmqm_training"

base_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/"
#base_dir = "/home/snw30/rds/hpc-work/3DMolecularDescriptors"

config_file = f"{base_dir}/data/{run}/dataset_config.yaml"
dataset_config = pyaml.parse_yaml_file_as(DatasetConfig, config_file)


model_dir = f"{base_dir}/transformer_model/{run}/"

pos_encoding_config = None

pos_encoding_config = RelativeDistancePositionalEncodingConfig(N_radial_basis_functions=16, distance_cutoff=32, d_projection=64,basis_function_type=RadialBasisFunctionType.BESSEL)

#pos_encoding_config = RandomWalkPositionalEncoding(k_hop_random_walk=16, d_projection=64)

embedding_preprocessor_config = EmbeddingPreprocessConfig(
    pseudoscalars=False, pseudoscalar_dimension=0,chiral_embedding_dimension=0
)

attention_layer_config = AttentionLayerConfig(
    num_heads=8,
    dim_feedforward=1024,
    dropout=0.3,
)

encoder_config = EncoderConfig(
    N_layers=4, attention_layer_config=attention_layer_config
)

decoder_config = None
decoder_config = DecoderConfig(N_layers=4, attention_layer_config=attention_layer_config)



mean_aggregator_config = MeanAggregatorConfig(aggregator_type= Aggregations.MEAN)

attention_aggregator_config = AttentionAggregatorConfig(aggregator_type= Aggregations.ATTENTION, num_heads = 8, attn_dropout= 0.3)

global_aggregator_config = GlobalAggregatorConfig(
    aggregator_type_config= attention_aggregator_config,
    global_molecular_descriptor_dropout=0.0
)

cf = ConfigFactory(
    dataset_config,
    embedding_preprocessor_config,
    attention_layer_config,
    encoder_config,
    global_aggregator_config,
    positional_encoding_config=pos_encoding_config,
    decoder_config = decoder_config
)





head_config_template = RegressionHeadConfig(
    activation_fn=torch.nn.SiLU(),
    hidden_dimensions=[512,256,128],
    head_type=HeadType.FULLY_CONNECTED,
)


os.makedirs(model_dir, exist_ok= True)

architecture_config = cf.create_architecture_config_template(
    model_directory=model_dir, head_config_template=head_config_template
)
