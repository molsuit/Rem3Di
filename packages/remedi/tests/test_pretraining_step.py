"""The denoising pretraining step: noise injection and the shared train/validation step."""

from __future__ import annotations

import torch
from torch import nn

from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.training.data.sample import PreprocessedSample, Sample
from remedi.training.pretraining import add_noise, denoising_step

EMBEDDING_DIMENSION = 4


class _Preprocessor(nn.Module):
    """Pads two molecules (3 and 2 atoms) into fixed embeddings."""

    def forward(self, sample: Sample) -> PreprocessedSample:
        embeddings = torch.arange(2 * 3 * EMBEDDING_DIMENSION, dtype=torch.float32)
        padding_mask = torch.tensor([[False, False, False], [False, False, True]])
        sample.padding_mask = padding_mask
        return PreprocessedSample(
            preprocessed_atomic_embeddings=embeddings.reshape(2, 3, -1) / 10.0,
            padding_mask=padding_mask,
            initial_pair_representation=torch.zeros(2, 3, 3, 1),
            geometrical_encoding=torch.zeros(2, 3, 3, 1),
            pair_mask=torch.zeros(2, 3, 3, dtype=torch.bool),
        )


class _Encoder(nn.Module):
    def forward(self, sample: PreprocessedSample) -> MolecularDescriptor:
        assert sample.preprocessed_atomic_embeddings is not None
        return MolecularDescriptor(
            tokens=sample.preprocessed_atomic_embeddings.mean(dim=1, keepdim=True)
        )


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)

    def forward(
        self, sample: PreprocessedSample, descriptor: MolecularDescriptor
    ) -> torch.Tensor:
        assert sample.preprocessed_atomic_embeddings is not None
        return self.linear(sample.preprocessed_atomic_embeddings) + descriptor.tokens


def _samples() -> Sample:
    return Sample(atomic_positions=torch.zeros(5, 3))


def _step(generator: torch.Generator | None = None):
    torch.manual_seed(0)
    decoder = _Decoder()
    return denoising_step(
        _samples(),
        preprocessor=_Preprocessor(),
        encoder=_Encoder(),
        decoder=decoder,
        noise_level=0.3,
        generator=generator,
    )


def test_add_noise_leaves_padding_untouched_and_follows_the_generator() -> None:
    embeddings = torch.ones(2, 3, EMBEDDING_DIMENSION)
    padding_mask = torch.tensor([[False, False, False], [False, True, True]])

    def noised(seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return add_noise(embeddings, padding_mask, 0.5, generator)

    first = noised(1)
    assert torch.equal(first, noised(1))
    assert not torch.equal(first, noised(2))
    assert torch.equal(first[padding_mask], embeddings[padding_mask])
    assert not torch.equal(first[~padding_mask], embeddings[~padding_mask])
    assert not first.requires_grad


def test_validation_loss_is_the_same_with_the_same_generator_seed() -> None:
    """Best-epoch selection compares validation losses: their noise must be fixed."""
    first = _step(torch.Generator().manual_seed(5))
    torch.randn(100)  # the global stream moves on between validation passes
    second = _step(torch.Generator().manual_seed(5))
    assert torch.equal(first.denoising_loss, second.denoising_loss)
    assert not torch.equal(
        first.denoising_loss, _step(torch.Generator().manual_seed(6)).denoising_loss
    )


def test_the_step_counts_atoms_and_backpropagates_into_the_decoder() -> None:
    result = _step()
    assert result.atom_count == 5
    assert result.molecular_descriptor.flat.shape == (2, EMBEDDING_DIMENSION)
    result.denoising_loss.backward()
