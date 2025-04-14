from importlib import resources

import pydantic_yaml as pyaml
import torch

from threedscriptors.configuration.architecture_config import (
    Activations,
    ArchitectureConfig,
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    RegressionHeadConfig,
)
from threedscriptors.configuration.config_utils import from_yaml
from threedscriptors.model.model_builder import ModelBuilder


def test_configuration():
    embedding_preprocessor_config = EmbeddingPreprocessConfig(
        input_irreps="10x0o", pseudoscalars=True, pseudoscalar_dimension=10
    )

    attention_layer_config = AttentionLayerConfig(
        input_dim=100,
        num_heads=8,
        dim_feedforward=512,
        embedding_dim=100,
        dropout=0.3,
    )

    encoder_config = EncoderConfig(
        N_layers=2, attention_layer_config=attention_layer_config
    )

    regression_head_config = RegressionHeadConfig(
        activation_fn=torch.nn.SiLU(),
        task_name="foo",
        hidden_dimensions=[512, 256, 128],
        input_dimensions=258,
    )

    global_aggregator_config = GlobalAggregatorConfig(
        input_dim=attention_layer_config.embedding_dim,
        aggregation_fn="mean",
    )

    architecture_configuration = ArchitectureConfig(
        embedding_preprocess_config=embedding_preprocessor_config,
        encoder_config=encoder_config,
        global_aggregator_config=global_aggregator_config,
        regression_head_config=regression_head_config,
    )

    config_str = pyaml.to_yaml_str(architecture_configuration)
    # print(config_str)
    reconstructed_architecture_config = pyaml.parse_yaml_raw_as(
        ArchitectureConfig, config_str
    )

    assert pyaml.to_yaml_str(reconstructed_architecture_config) == config_str


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


def test_model_reconstruction():
    yaml_file = resources.files("tests") / "test_architecture_config.yaml"

    architecture_config = from_yaml(yaml_file, ArchitectureConfig)

    print(architecture_config)
    builder = ModelBuilder(architecture_config)

    try:
        _ = builder.build_model()
    except Exception as err:
        raise AssertionError from err


def test_model_construction_from_yaml():
    yaml_file = resources.files("tests") / "test_architecture_config.yaml"

    architecture_config = from_yaml(yaml_file, ArchitectureConfig)

    try:
        _ = ModelBuilder(architecture_config=architecture_config).build_model()
    except Exception as err:
        raise AssertionError from err
