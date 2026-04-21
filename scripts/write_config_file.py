import os

import pydantic_yaml as pyaml

from threedscriptors.configuration.architecture_config import (
    AttentionLayerConfig,
    BesselBasisConfig,
    DecoderConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    EncoderDecoderArchitectureConfig,
    GlobalAggregatorConfig,
    PMAAggregatorConfig,
    RelativeDistancePositionalEncodingConfig,
)
from threedscriptors.configuration.dataset_config import DatasetConfig

run = "pcqm"

base_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/"

config_file = f"{base_dir}/datasets/{run}/dataset_config.yaml"
dataset_config = pyaml.parse_yaml_file_as(DatasetConfig, config_file)


model_dir = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/training_runs/pcqm_with_pma_agg"


pos_encoding_config = RelativeDistancePositionalEncodingConfig(
    N_radial_basis_functions=16,
    distance_cutoff=32,
    d_projection=64,
    basis_function_config=BesselBasisConfig(),
)

embedding_preprocessor_config = EmbeddingPreprocessConfig(
    input_irreps=dataset_config.irreps,
    pseudoscalars=True,
    pseudoscalar_dimension=64,
    chiral_embedding_dimension=64,
)

attention_layer_config = AttentionLayerConfig(
    num_heads=8,
    dim_feedforward=1024,
    dropout=0.3,
)

encoder_config = EncoderConfig(
    N_layers=4, attention_layer_config=attention_layer_config
)

decoder_config = DecoderConfig(
    N_layers=4, attention_layer_config=attention_layer_config
)

pma_aggregator_config = PMAAggregatorConfig(head_dim=320, num_seeds=2)

global_aggregator_config = GlobalAggregatorConfig(
    aggregator_type_config=pma_aggregator_config,
    output_dim=320,
    global_molecular_descriptor_dropout=0.0,
)


os.makedirs(model_dir, exist_ok=True)

architecture_config = EncoderDecoderArchitectureConfig(
    embedding_preprocess_config=embedding_preprocessor_config,
    encoder_config=encoder_config,
    global_aggregator_config=global_aggregator_config,
    positional_encoding_config=pos_encoding_config,
    decoder_config=decoder_config,
)

pyaml.to_yaml_file(f"{model_dir}/architecture_config.yaml", architecture_config)
