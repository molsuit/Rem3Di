from importlib import resources

import pydantic_yaml as pyaml
import pytest
import torch

from threedscriptors.configuration.architecture_config import (
    Activations,
    AttentionLayerConfig,
    DecoderConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    EncoderDecoderArchitectureConfig,
    GlobalAggregatorConfig,
    MeanAggregatorConfig,
    RegressionArchitectureConfig,
    RegressionHeadConfig,
    RelativeDistancePositionalEncodingConfig,
)
from threedscriptors.configuration.config_utils import from_yaml


def _minimal_shared_parts(fill_derived: bool):
    embedding_preprocessor_config = EmbeddingPreprocessConfig(
        input_irreps="128x0e+128x1o+128x0e",
        pseudoscalars=False,
        pseudoscalar_dimension=0,
        chiral_embedding_dimension=16,
    )

    # output_irreps_dim of this preprocess config is 256 (pseudoscalars=False ->
    # only even invariants -> 128 + 128).
    attention_layer_config = AttentionLayerConfig(
        num_heads=8,
        dim_feedforward=512,
        dropout=0.3,
        embedding_dim=256 if fill_derived else None,
    )

    encoder_config = EncoderConfig(
        N_layers=2,
        attention_layer_config=attention_layer_config,
        d_pair=64 if fill_derived else None,
        d_geo=16 if fill_derived else None,
    )

    global_aggregator_config = GlobalAggregatorConfig(
        aggregator_type_config=MeanAggregatorConfig(),
        input_dim=256 if fill_derived else None,
        output_dim=256 if fill_derived else None,
    )

    positional_encoding_config = RelativeDistancePositionalEncodingConfig(
        N_radial_basis_functions=16,
        d_projection=64,
        distance_cutoff=32.0,
    )

    return (
        embedding_preprocessor_config,
        encoder_config,
        global_aggregator_config,
        positional_encoding_config,
    )


def _build_minimal_regression_config(
    fill_derived: bool = False,
) -> RegressionArchitectureConfig:
    (
        embedding_preprocessor_config,
        encoder_config,
        global_aggregator_config,
        positional_encoding_config,
    ) = _minimal_shared_parts(fill_derived)

    regression_head_config = RegressionHeadConfig(
        activation_fn=torch.nn.SiLU(),
        task_name="foo",
        hidden_dimensions=[512, 256, 128],
        input_dimensions=256 if fill_derived else None,
    )

    return RegressionArchitectureConfig(
        embedding_preprocess_config=embedding_preprocessor_config,
        encoder_config=encoder_config,
        global_aggregator_config=global_aggregator_config,
        regression_head_config=[regression_head_config],
        positional_encoding_config=positional_encoding_config,
    )


def _build_minimal_encoder_decoder_config(
    fill_derived: bool = False,
) -> EncoderDecoderArchitectureConfig:
    (
        embedding_preprocessor_config,
        encoder_config,
        global_aggregator_config,
        positional_encoding_config,
    ) = _minimal_shared_parts(fill_derived)

    decoder_config = DecoderConfig(
        N_layers=2,
        attention_layer_config=AttentionLayerConfig(
            num_heads=8,
            dim_feedforward=512,
            dropout=0.3,
            embedding_dim=256 if fill_derived else None,
        ),
        d_descriptor=256 if fill_derived else None,
        d_pair=64 if fill_derived else None,
        d_geo=16 if fill_derived else None,
    )

    return EncoderDecoderArchitectureConfig(
        embedding_preprocess_config=embedding_preprocessor_config,
        encoder_config=encoder_config,
        global_aggregator_config=global_aggregator_config,
        positional_encoding_config=positional_encoding_config,
        decoder_config=decoder_config,
    )


def test_regression_yaml_roundtrip():
    cfg = _build_minimal_regression_config(fill_derived=True)
    config_str = pyaml.to_yaml_str(cfg)
    reconstructed = pyaml.parse_yaml_raw_as(RegressionArchitectureConfig, config_str)
    assert pyaml.to_yaml_str(reconstructed) == config_str


def test_encoder_decoder_yaml_roundtrip():
    cfg = _build_minimal_encoder_decoder_config(fill_derived=True)
    config_str = pyaml.to_yaml_str(cfg)
    reconstructed = pyaml.parse_yaml_raw_as(
        EncoderDecoderArchitectureConfig, config_str
    )
    assert pyaml.to_yaml_str(reconstructed) == config_str


def test_function_reconstruction():
    regression_head_config_0 = RegressionHeadConfig(
        activation_fn=torch.nn.SiLU(),
        task_name="foo",
        hidden_dimensions=[512, 256, 128],
        input_dimensions=258,
    )

    regression_head_config_1 = RegressionHeadConfig(
        activation_fn="silu",
        task_name="foo",
        hidden_dimensions=[512, 256, 128],
        input_dimensions=258,
    )

    regression_head_config_2 = RegressionHeadConfig(
        activation_fn=Activations.SILU,
        task_name="foo",
        hidden_dimensions=[512, 256, 128],
        input_dimensions=258,
    )

    config_str = pyaml.to_yaml_str(regression_head_config_0)

    assert config_str == pyaml.to_yaml_str(
        regression_head_config_1
    ) and config_str == pyaml.to_yaml_str(regression_head_config_2)

    assert regression_head_config_0.activation_fn(torch.Tensor([0.0])) == 0.0


def test_regression_cascade_fills_derived_dims():
    cfg = _build_minimal_regression_config(fill_derived=False)

    embed_dim = cfg.embedding_preprocess_config.output_irreps_dim
    assert embed_dim == 256

    assert cfg.encoder_config.attention_layer_config.embedding_dim == embed_dim
    assert cfg.encoder_config.d_pair == cfg.positional_encoding_config.d_projection
    assert (
        cfg.encoder_config.d_geo
        == cfg.positional_encoding_config.N_radial_basis_functions
    )

    assert cfg.global_aggregator_config.input_dim == embed_dim
    assert cfg.global_aggregator_config.output_dim == embed_dim

    assert (
        cfg.regression_head_config[0].input_dimensions
        == cfg.global_aggregator_config.output_dim
    )


def test_encoder_decoder_cascade_fills_derived_dims():
    cfg = _build_minimal_encoder_decoder_config(fill_derived=False)

    embed_dim = cfg.embedding_preprocess_config.output_irreps_dim
    assert embed_dim == 256

    assert cfg.encoder_config.attention_layer_config.embedding_dim == embed_dim
    assert cfg.global_aggregator_config.input_dim == embed_dim

    assert cfg.decoder_config.attention_layer_config.embedding_dim == embed_dim
    assert cfg.decoder_config.d_descriptor == cfg.global_aggregator_config.output_dim
    assert cfg.decoder_config.d_pair == cfg.positional_encoding_config.d_projection
    assert (
        cfg.decoder_config.d_geo
        == cfg.positional_encoding_config.N_radial_basis_functions
    )


def test_cascade_preserves_explicit_overrides():
    # Explicit dims must survive the cascade unchanged, so users can override.
    cfg = _build_minimal_regression_config(fill_derived=False)
    cfg_override = cfg.model_copy(deep=True)
    cfg_override.encoder_config.d_pair = 128
    cfg_override = RegressionArchitectureConfig.model_validate(
        cfg_override.model_dump()
    )
    assert cfg_override.encoder_config.d_pair == 128


def test_regression_yaml_parses_against_current_schema():
    yaml_file = resources.files("tests") / "test_architecture_config.yaml"
    cfg = from_yaml(yaml_file, RegressionArchitectureConfig)
    assert isinstance(cfg, RegressionArchitectureConfig)
    # Derived dims should be filled by the cascade even though the YAML omits them.
    assert cfg.encoder_config.attention_layer_config.embedding_dim == 256
    assert cfg.global_aggregator_config.input_dim == 256


def test_encoder_decoder_yaml_parses_against_current_schema():
    yaml_file = (
        resources.files("tests") / "test_architecture_config_encoder_decoder.yaml"
    )
    cfg = from_yaml(yaml_file, EncoderDecoderArchitectureConfig)
    assert isinstance(cfg, EncoderDecoderArchitectureConfig)
    assert cfg.encoder_config.attention_layer_config.embedding_dim == 256
    assert cfg.decoder_config.d_descriptor == 256


def test_model_reconstruction():
    # Full build touches the MACE-integrated preprocessor, which imports
    # torch_sim. Skip when unavailable so the schema tests above remain green.
    pytest.importorskip("torch_sim")

    yaml_file = resources.files("tests") / "test_architecture_config.yaml"
    cfg = from_yaml(yaml_file, RegressionArchitectureConfig)
    _ = cfg.build()
