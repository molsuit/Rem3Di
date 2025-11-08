import pytest
import torch

from threedscriptors.configuration.architecture_config import EmbeddingPreprocessConfig
from threedscriptors.model.preprocessing.atomic_descriptor_preprocessor import (
    AtomicDescriptorPreprocessor,
)


def test_overwrite_mean_atomic_embeddings():
    config = EmbeddingPreprocessConfig(
        pseudoscalars=True,
        input_irreps="128x0e+128x1o+128x0e",
        pseudoscalar_dimension=128,
        input_embedding_size=384,
    )

    preprocessor = AtomicDescriptorPreprocessor(config)

    mean = torch.ones(size=(384,))
    std = torch.ones(size=(384,))

    preprocessor.register_embedding_normalization(mean, std)

    with pytest.raises(AssertionError):
        # reregistering the embedding normalization should fail, becuase it is not permissible to overwrite this. Raising this error stops the user from changing the normalization factors after the model has been trained with for the original normlaization factors
        preprocessor.register_embedding_normalization(mean, std)


def test_reload_embedding_normalization(tmp_path):
    config = EmbeddingPreprocessConfig(
        pseudoscalars=False,
        input_irreps="128x0e+128x1o+128x0e",
        pseudoscalar_dimension=128,
        input_embedding_size=384,
    )

    preprocessor = AtomicDescriptorPreprocessor(config)

    mean = 2 * torch.ones(size=(1,1,384))
    std = 2 * torch.ones(size=(1,1,384))

    preprocessor.register_embedding_normalization(mean, std)

    file = tmp_path / "preprocessor.pth"

    torch.save(preprocessor.state_dict(), file)

    reloaded_preprocessor_state_dict = torch.load(file)

    reloaded_preprocessor = AtomicDescriptorPreprocess(config)

    reloaded_preprocessor.load_state_dict(
        reloaded_preprocessor_state_dict, strict=False
    )

    assert torch.all(reloaded_preprocessor.get_buffer("mean_atomic_embedding") == mean)
    assert torch.all(reloaded_preprocessor.get_buffer("std_atomic_embedding") == std)
