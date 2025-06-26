import torch

from threedscriptors.configuration.architecture_config import (
    AttentionLayerConfig,
    EmbeddingPreprocessConfig,
    EncoderConfig,
    GlobalAggregatorConfig,
    HeadType,
    PositionalEncodingConfig,
    RegressionHeadConfig,
)
from threedscriptors.data_handling.data_build_pipeline import (
    AtomicPositionsStage,
    PipelineOrchestrator,
    ReloadFromDiskStage,
)
from threedscriptors.model.atomic_descriptor_preprocess import InvariantsFilter
from threedscriptors.model.global_aggregator import GlobalAggregator
from threedscriptors.model.regression_models import (
    MultitaskHeads,
    StructureBasedMultitaskRegressionModel,
    TransformerPairEncoder,
)
from threedscriptors.model.structural_encoding import PairDistanceMatrixEncodingBlock

pos_encoding_config = PositionalEncodingConfig(
    N_radial_basis_functions=16, distance_cutoff=20.0, d_projection=64
)
structure_encoding_block = PairDistanceMatrixEncodingBlock(
    **pos_encoding_config.model_dump()
)


attn_layer_config = AttentionLayerConfig(
    num_heads=8, dim_feedforward=1024, dropout=0.2, embedding_dim=256
)

head_config_template = [RegressionHeadConfig(
    activation_fn=torch.nn.SiLU(),
    hidden_dimensions=[256,128],
    head_type=HeadType.FULLY_CONNECTED,
    task_name="test",
    input_dimensions=256
)]

multitask_heads = MultitaskHeads(head_config_template)

encoder_config = EncoderConfig(N_layers=5, attention_layer_config=attn_layer_config)

pair_encoder = TransformerPairEncoder(encoder_config=encoder_config, d_pair=64)


embedding_preprocessor_config = EmbeddingPreprocessConfig(
    pseudoscalars=False,
    pseudoscalar_dimension=0,
    pseudoscalar_embedding_dim=0,
    input_irreps="128x0e+128x1o+128x0e",
    output_irreps="128x0e+128x0e",
    input_embedding_size=640,
)


preprocessor = InvariantsFilter(embedding_preprocessor_config)


gl_agg_config = GlobalAggregatorConfig(
    aggregation_fn="mean", input_dim=256, output_dim=256
)
global_aggregator = GlobalAggregator(gl_agg_config)

model = StructureBasedMultitaskRegressionModel(
    structure_encoding_block=structure_encoding_block,
    pair_encoder=pair_encoder,
    global_aggregator=global_aggregator,
    preprocessor=preprocessor,
    multitask_heads = multitask_heads
)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))




stages = [ReloadFromDiskStage("/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/embedding_w_pos_test"), AtomicPositionsStage()]

dataset = PipelineOrchestrator(stages).build()
dataset.to_torch()



sample = dataset[1]
print(sample.padding_mask)
sample.padding_mask = sample.padding_mask[None, :].bool()
sample.atomic_positions = sample.atomic_positions[None, :, :]
sample.embeddings = sample.embeddings[None, :, :]

descriptor = model(sample)
