from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocessor,
)
from threedscriptors.model.model_builder import ModelBuilder
from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig

import torch 

path_to_load_model = "/share/snw30/projects/chiral_mols/training_runs/46-2025_07_26_12_47_35-PretrainingForCMRT/chiral_embedding_model.pth"
config = EmbeddingPreprocessConfig(
    chiral_embedding_dimension=32,
    input_dimension=640,
    input_equivariant_dimension=384,
    input_invariant_dimension=256,
    input_irreps="128x0e+128x1o+128x0e",
    output_irreps="128x0e+128x0e+32x0o",
    output_irreps_dim=288,
    pseudoscalar_dimension=64,
    pseudoscalar_irrep="64x0o",
    pseudoscalars=True,
    reload_state_dict=path_to_load_model,
)
mb = ModelBuilder(architecture_config=None)

embedding_model: AtomicDescriptorPreprocessor = mb.build_atomic_preprocessor(config)


# embedding_model takes atomic descriptors of B,N_max_atoms,input_dimension and padding masks (B, N_max_atoms): True for padding, False for real atoms. embedding_model returns a sample = PreprocessedSample object with the concatenated invariant+chiral embeddings on the sample.preprocessed_atomic_embeddings

B = 5
N_max_atoms = 10
F = config.input_dimension

embeddings = torch.rand(B,N_max_atoms,F)
padding_masks = torch.randint(0, 2, (B, N_max_atoms), dtype=torch.bool)


embeddings = embedding_model(embeddings, padding_masks)
