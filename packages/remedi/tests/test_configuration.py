from pathlib import Path

import pytest
import torch
import yaml
from pydantic import TypeAdapter, ValidationError

from remedi.configuration.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
    DecoderConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    EncoderDecoderArchitectureConfig,
    EncoderOnlyArchitectureConfig,
    GlobalAggregatorConfig,
    MeanAggregatorConfig,
    RegressionArchitectureConfig,
    RegressionHeadConfig,
    RelativeDistancePositionalEncodingConfig,
)
from remedi.configuration.mace_config import MaceConfig
from remedi_data.yaml_io import read_model, yaml_text

TESTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = TESTS_DIR.parents[2]


def _minimal_shared_parts():
    embedding_preprocessor_config = EmbeddingPreprocessConfig(
        mace_per_layer_irreps=["128x0e+128x1o", "128x0e"],
        pseudoscalars=False,
        pseudoscalar_dimension=0,
        chiral_embedding_dimension=16,
    )

    encoder_config = EncoderConfig(
        N_layers=2,
        embedding_dimension=96,
        attention_layer_config=AttentionLayerConfig(
            num_heads=8, dim_feedforward=512, dropout=0.3
        ),
    )

    global_aggregator_config = GlobalAggregatorConfig(
        aggregator_config=MeanAggregatorConfig(),
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


def _build_minimal_regression_config() -> RegressionArchitectureConfig:
    (
        embedding_preprocessor_config,
        encoder_config,
        global_aggregator_config,
        positional_encoding_config,
    ) = _minimal_shared_parts()

    return RegressionArchitectureConfig(
        embedding_preprocess_config=embedding_preprocessor_config,
        encoder_config=encoder_config,
        global_aggregator_config=global_aggregator_config,
        regression_head_config=[
            RegressionHeadConfig(task_name="foo", hidden_dimensions=[512, 256, 128])
        ],
        positional_encoding_config=positional_encoding_config,
    )


def _build_minimal_encoder_decoder_config() -> EncoderDecoderArchitectureConfig:
    (
        embedding_preprocessor_config,
        encoder_config,
        global_aggregator_config,
        positional_encoding_config,
    ) = _minimal_shared_parts()

    decoder_config = DecoderConfig(
        N_layers=2,
        embedding_dimension=48,
        attention_layer_config=AttentionLayerConfig(
            num_heads=8, dim_feedforward=512, dropout=0.3
        ),
    )

    return EncoderDecoderArchitectureConfig(
        embedding_preprocess_config=embedding_preprocessor_config,
        encoder_config=encoder_config,
        global_aggregator_config=global_aggregator_config,
        positional_encoding_config=positional_encoding_config,
        decoder_config=decoder_config,
    )


def test_regression_yaml_roundtrip():
    cfg = _build_minimal_regression_config()
    config_str = yaml_text(cfg.model_dump(mode="json"))
    reconstructed = RegressionArchitectureConfig.model_validate(
        yaml.safe_load(config_str)
    )
    assert yaml_text(reconstructed.model_dump(mode="json")) == config_str


def test_encoder_decoder_yaml_roundtrip():
    cfg = _build_minimal_encoder_decoder_config()
    config_str = yaml_text(cfg.model_dump(mode="json"))
    reconstructed = EncoderDecoderArchitectureConfig.model_validate(
        yaml.safe_load(config_str)
    )
    assert yaml_text(reconstructed.model_dump(mode="json")) == config_str


@pytest.mark.parametrize("activation", ["silu", "gelu", "relu", "tanh"])
def test_activation_yaml_roundtrip(activation):
    # Regression: "GELU" mapped to the missing torch.nn.gelu, and a callable
    # activation dumped as its class name and did not reload.
    head = RegressionHeadConfig(task_name="foo", activation=activation)
    config_str = yaml_text(head.model_dump(mode="json"))
    reconstructed = RegressionHeadConfig.model_validate(yaml.safe_load(config_str))
    assert reconstructed == head


def test_unknown_activation_rejected():
    with pytest.raises(ValidationError):
        RegressionHeadConfig(task_name="foo", activation="swish")


def test_dimensions_are_derived():
    cfg = _build_minimal_regression_config()
    dimensions = cfg.dimensions

    # pseudoscalars=False -> only the even invariants: 128 + 128. They are
    # both what the encoder reads and what the decoder reconstructs.
    assert dimensions.atomic_descriptor_dimension == 256
    assert dimensions.target_dimension == 256
    # The encoder width is configured, not tied to the descriptor width.
    assert dimensions.embedding_dimension == 96
    assert dimensions.pair_dimension == 64
    assert dimensions.radial_basis_dimension == 16
    # output_dim None -> descriptor width is the encoder width.
    assert dimensions.descriptor_dimension == 96
    assert dimensions.descriptor_flat_dimension == 96


def test_the_chiral_embedding_is_encoder_input_but_not_denoising_target():
    data = _build_minimal_encoder_decoder_config().model_dump(mode="json")
    data["embedding_preprocess_config"]["mace_layer_indices"] = [0]
    data["embedding_preprocess_config"]["pseudoscalars"] = True
    data["embedding_preprocess_config"]["pseudoscalar_dimension"] = 8
    dimensions = EncoderDecoderArchitectureConfig.model_validate(data).dimensions
    assert dimensions.atomic_descriptor_dimension == 128 + 16
    assert dimensions.target_dimension == 128


def test_denoising_needs_fixed_normalisation_statistics():
    data = _build_minimal_encoder_decoder_config().model_dump(mode="json")
    data["embedding_preprocess_config"]["invariant_normalization_config"] = {
        "kind": "on_the_fly_normalization",
        "momentum": 0.1,
        "warm_up_batches": 10,
    }
    with pytest.raises(ValidationError, match="move the denoising target"):
        EncoderDecoderArchitectureConfig.model_validate(data)


def test_the_encoder_and_decoder_have_their_own_widths():
    bundle_config = _build_minimal_encoder_decoder_config()
    decoder = bundle_config.decoder_config.build(bundle_config.dimensions)
    encoder = bundle_config._build_encoder()
    assert encoder.input_projection.out_features == 96
    assert decoder.input_projection.in_features == 256
    assert decoder.input_projection.out_features == 48
    assert decoder.output_head.out_features == 256


@pytest.mark.parametrize(
    "section, key, value",
    [
        ("encoder_config", "d_pair", 128),
        ("encoder_config", "d_geo", 8),
        ("global_aggregator_config", "input_dim", 128),
        ("embedding_preprocess_config", "input_irreps", "128x0e"),
        ("embedding_preprocess_config", "equivariant_rms_normalization", True),
        ("embedding_preprocess_config", "output_irreps_dim", 256),
    ],
)
def test_derived_or_removed_keys_are_rejected(section, key, value):
    # Regression: dimension fields that must equal derived values were settable
    # and unchecked (a mismatch validated and failed in forward), and computed
    # fields were dumped into yaml and ignored on reload.
    data = _build_minimal_encoder_decoder_config().model_dump(mode="json")
    data[section][key] = value
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EncoderDecoderArchitectureConfig.model_validate(data)


def test_decoder_dims_are_rejected():
    data = _build_minimal_encoder_decoder_config().model_dump(mode="json")
    data["decoder_config"]["d_descriptor"] = 64
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EncoderDecoderArchitectureConfig.model_validate(data)
    data = _build_minimal_encoder_decoder_config().model_dump(mode="json")
    data["decoder_config"]["attention_layer_config"]["embedding_dim"] = 128
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EncoderDecoderArchitectureConfig.model_validate(data)


def test_mistyped_key_is_rejected():
    data = _build_minimal_regression_config().model_dump(mode="json")
    data["positional_encoding_config"]["distance_cuttoff"] = 5.0
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RegressionArchitectureConfig.model_validate(data)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MaceConfig.model_validate({"devise": "cpu"})


def test_dump_has_no_computed_fields():
    dumped = _build_minimal_encoder_decoder_config().model_dump(mode="json")
    embedding = dumped["embedding_preprocess_config"]
    for computed in [
        "input_irreps",
        "output_irreps",
        "output_irreps_dim",
        "invariant_irreps",
        "pseudoscalar_irrep",
        "input_dimension",
    ]:
        assert computed not in embedding
    assert "descriptor_flat_dim" not in dumped["global_aggregator_config"]
    assert "descriptor_seq_len" not in dumped["global_aggregator_config"]


def test_precomputed_embeddings_need_per_layer_irreps():
    data = _build_minimal_regression_config().model_dump(mode="json")
    data["embedding_preprocess_config"]["mace_per_layer_irreps"] = None
    with pytest.raises(ValidationError, match="mace_per_layer_irreps is not set"):
        RegressionArchitectureConfig.model_validate(data)


def test_mace_per_layer_irreps_checked_against_mace(monkeypatch):
    # Regression: input_irreps was never checked against the MACE model.
    from e3nn.o3 import Irreps

    monkeypatch.setattr(
        MaceConfig,
        "get_per_layer_irreps",
        lambda self: [Irreps("128x0e+128x1o"), Irreps("128x0e")],
    )
    data = _build_minimal_regression_config().model_dump(mode="json")
    data["mace_config"] = {"device": "cpu"}
    RegressionArchitectureConfig.model_validate(data)  # matches: fine

    data["embedding_preprocess_config"]["mace_per_layer_irreps"] = ["64x0e"]
    with pytest.raises(ValidationError, match="does not match the configured MACE"):
        RegressionArchitectureConfig.model_validate(data)

    data["embedding_preprocess_config"]["mace_per_layer_irreps"] = None
    filled = RegressionArchitectureConfig.model_validate(data)
    assert filled.embedding_preprocess_config.input_irreps == Irreps(
        "128x0e+128x1o+128x0e"
    )


def test_regression_yaml_parses_against_current_schema():
    yaml_file = TESTS_DIR / "test_architecture_config.yaml"
    cfg = read_model(yaml_file, RegressionArchitectureConfig)
    assert isinstance(cfg, RegressionArchitectureConfig)
    assert cfg.dimensions.embedding_dimension == 256


def test_encoder_decoder_yaml_parses_against_current_schema():
    yaml_file = TESTS_DIR / "test_architecture_config_encoder_decoder.yaml"
    cfg = read_model(yaml_file, EncoderDecoderArchitectureConfig)
    assert isinstance(cfg, EncoderDecoderArchitectureConfig)
    assert cfg.dimensions.embedding_dimension == 256
    assert cfg.dimensions.descriptor_dimension == 256
    assert cfg.decoder_config.embedding_dimension == 128


def test_encoder_only_from_encoder_yaml_strips_decoder(tmp_path):
    src = TESTS_DIR / "test_architecture_config_encoder_decoder.yaml"
    target = tmp_path / "post_training_architecture_config.yaml"
    target.write_bytes(src.read_bytes())

    cfg = EncoderOnlyArchitectureConfig.from_encoder_yaml(tmp_path)

    assert isinstance(cfg, EncoderOnlyArchitectureConfig)
    assert cfg.kind == "encoder_only"
    assert cfg.dimensions.embedding_dimension == 256


SHIPPED_ARCHITECTURE_CONFIGS = sorted(
    [
        *(REPOSITORY_ROOT / "configs" / "training").rglob("architecture_config.yaml"),
        *(REPOSITORY_ROOT / "training_runs").rglob("architecture_config.yaml"),
    ]
)


@pytest.mark.parametrize(
    "path",
    SHIPPED_ARCHITECTURE_CONFIGS,
    ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)),
)
def test_shipped_architecture_configs_validate(path, monkeypatch):
    # The shipped configs name machine-specific MACE-POLAR checkpoints; stand in
    # its per-layer irreps so the schema (extra="forbid") is what is checked.
    from e3nn.o3 import Irreps

    monkeypatch.setattr(
        MaceConfig,
        "get_per_layer_irreps",
        lambda self: [Irreps("512x0e+512x1o"), Irreps("512x0e+512x1o")],
    )
    data = yaml.safe_load(path.read_text())
    cfg = TypeAdapter(ArchitectureConfig).validate_python(data)
    assert cfg.dimensions.embedding_dimension > 0


def test_model_reconstruction():
    yaml_file = TESTS_DIR / "test_architecture_config.yaml"
    cfg = read_model(yaml_file, RegressionArchitectureConfig)
    model = cfg.build()
    assert isinstance(model, torch.nn.Module)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
