from mace.calculators import MACECalculator
from torch import nn

from threedscriptors.configuration.architecture_config import (
    ArchitectureConfig,
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    RegressionHeadConfig,
)
from threedscriptors.configuration.config_utils import to_yaml
from threedscriptors.model.atomic_descriptor_preprocess import (
    InvariantsFilter,
    PseudoscalarGenerator,
)
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.utils.model_utils import get_mace_calculator_irrep_signature

mace_model_path = (
    "/data/fast-pc-06/snw30/projects/models/2023-12-03-mace-128-L1_epoch-199.model"
)


device = "cuda"

mace_calculator = MACECalculator(
    model_paths=mace_model_path, device=device, enable_cueq=True
)

calculator_irreps = get_mace_calculator_irrep_signature(mace_calculator)


embedding_preprocessor_config = EmbeddingPreprocessConfig(
    input_irreps=calculator_irreps, pseudoscalars=True
)

if embedding_preprocessor_config.pseudoscalars:
    preprocessor = PseudoscalarGenerator(embedding_preprocessor_config)
else:
    preprocessor = InvariantsFilter(embedding_preprocessor_config)


attention_layer_config = AttentionLayerConfig(
    input_dim=preprocessor.config.output_irreps_dim,
    num_heads=8,
    dim_feedforward=512,
    embedding_dim=preprocessor.config.output_irreps_dim,
    dropout=0.3,
)

encoder_config = EncoderConfig(
    N_layers=2, attention_layer_config=attention_layer_config
)

regression_head_config_0 = RegressionHeadConfig(
    task_name="foo",
    activation_fn=nn.SiLU(),
    hidden_dimensions=[512, 256, 128],
    input_dimensions=256,
)


regression_head_config_1 = RegressionHeadConfig(
    task_name="bar",
    activation_fn=nn.SiLU(),
    hidden_dimensions=[512, 256, 128],
    input_dimensions=256,
)

global_aggregator_config = GlobalAggregatorConfig(
    input_dim=attention_layer_config.embedding_dim,
    aggregation_fn="mean",
)

global_aggregator = GlobalAggregator(global_aggregator_config)

architecture_config = ArchitectureConfig(
    embedding_preprocess_config=embedding_preprocessor_config,
    encoder_config=encoder_config,
    global_aggregator_config=global_aggregator_config,
    regression_head_config=[regression_head_config_0, regression_head_config_1],
)


to_yaml("test.yaml", architecture_config)
