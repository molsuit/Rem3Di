import os
from pathlib import Path

import pydantic_yaml as pyaml

from remedi.configuration.architecture_config import (
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
from remedi.configuration.mace_config import MaceConfig

model_dir = "/path/to/projects/mol_descriptors/training_runs"


pos_encoding_config = RelativeDistancePositionalEncodingConfig(
    N_radial_basis_functions=16,
    distance_cutoff=32,
    d_projection=64,
    basis_function_config=BesselBasisConfig(),
)

mace_config = MaceConfig(
    model_path=Path(
        "/path/to/projects/mol_descriptors/mace_model/MACE-POLAR-1-M.model"
    ),
)

embedding_preprocessor_config = EmbeddingPreprocessConfig(
    pseudoscalars=False,
    pseudoscalar_dimension=64,
    chiral_embedding_dimension=64,
)

attention_layer_config = AttentionLayerConfig(
    num_heads=8,
    dim_feedforward=2048,
    dropout=0.3,
)

encoder_config = EncoderConfig(
    N_layers=4, attention_layer_config=attention_layer_config
)

decoder_config = DecoderConfig(
    N_layers=4, attention_layer_config=attention_layer_config
)

pma_aggregator_config = PMAAggregatorConfig(head_dim=64, num_heads=4, num_seeds=2)

global_aggregator_config = GlobalAggregatorConfig(
    aggregator_type_config=pma_aggregator_config,
    output_dim=320,
    global_molecular_descriptor_dropout=0.0,
)


os.makedirs(model_dir, exist_ok=True)

architecture_config = EncoderDecoderArchitectureConfig(
    mace_config=mace_config,
    embedding_preprocess_config=embedding_preprocessor_config,
    encoder_config=encoder_config,
    global_aggregator_config=global_aggregator_config,
    positional_encoding_config=pos_encoding_config,
    decoder_config=decoder_config,
)

pyaml.to_yaml_file(f"{model_dir}/architecture_config.yaml", architecture_config)
