"""The denoising pretraining step: noise injection, the fixed target and the
preconditioned loss shared by training and validation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from remedi.configuration.architecture_config import (
    AttentionLayerConfig,
    DecoderConfig,
    EncoderConfig,
    ModelDimensions,
)
from remedi.configuration.training_config import (
    FixedNoiseLevelConfig,
    LogNormalNoiseLevelConfig,
)
from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.model.pooling import MeanPool
from remedi.training.data.sample import PreprocessedSample, Sample
from remedi.training.pretraining import (
    add_noise,
    denoising_step,
    descriptor_statistics,
)

TARGET_DIMENSION = 4
PADDING_MASK = torch.tensor([[False, False, False], [False, False, True]])


def _target() -> torch.Tensor:
    target = torch.arange(2 * 3 * TARGET_DIMENSION, dtype=torch.float32)
    target = target.reshape(2, 3, -1) / 10.0
    return target.masked_fill(PADDING_MASK[:, :, None], 0.0)


class _Preprocessor(nn.Module):
    """Pads two molecules (3 and 2 atoms) into fixed normalised invariants."""

    def forward(self, sample: Sample) -> PreprocessedSample:
        sample.padding_mask = PADDING_MASK
        return PreprocessedSample(
            preprocessed_atomic_embeddings=_target(),
            normalized_invariants=_target(),
            padding_mask=PADDING_MASK,
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
    """Records what it is given; ``zero`` makes it output zero (as the real
    decoder does at initialisation)."""

    def __init__(self, zero: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(TARGET_DIMENSION, TARGET_DIMENSION)
        self.zero = zero
        self.received: PreprocessedSample | None = None
        self.noise_level: torch.Tensor | None = None

    def forward(
        self,
        sample: PreprocessedSample,
        descriptor: MolecularDescriptor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        assert sample.preprocessed_atomic_embeddings is not None
        self.received = sample
        self.noise_level = noise_level
        output = self.linear(sample.preprocessed_atomic_embeddings) + descriptor.tokens
        return torch.zeros_like(output) if self.zero else output


def _step(
    generator: torch.Generator | None = None,
    noise=None,
    decoder: _Decoder | None = None,
):
    torch.manual_seed(0)
    return denoising_step(
        Sample(atomic_positions=torch.zeros(5, 3)),
        preprocessor=_Preprocessor(),
        encoder=_Encoder(),
        decoder=decoder if decoder is not None else _Decoder(),
        noise=noise if noise is not None else LogNormalNoiseLevelConfig(),
        generator=generator,
    )


def test_add_noise_leaves_padding_untouched_and_follows_the_generator() -> None:
    embeddings = torch.ones(2, 3, TARGET_DIMENSION)
    padding_mask = torch.tensor([[False, False, False], [False, True, True]])

    def noised(seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return add_noise(embeddings, padding_mask, torch.tensor([0.5, 0.5]), generator)

    first = noised(1)
    assert torch.equal(first, noised(1))
    assert not torch.equal(first, noised(2))
    assert torch.equal(first[padding_mask], embeddings[padding_mask])
    assert not torch.equal(first[~padding_mask], embeddings[~padding_mask])
    assert not first.requires_grad


def test_add_noise_scales_the_noise_per_molecule() -> None:
    embeddings = torch.zeros(2, 200, TARGET_DIMENSION)
    padding_mask = torch.zeros(2, 200, dtype=torch.bool)
    noised = add_noise(
        embeddings,
        padding_mask,
        torch.tensor([0.1, 10.0]),
        torch.Generator().manual_seed(0),
    )
    assert noised[0].std().item() == pytest.approx(0.1, rel=0.1)
    assert noised[1].std().item() == pytest.approx(10.0, rel=0.1)


def test_validation_loss_is_the_same_with_the_same_generator_seed() -> None:
    """Best-epoch selection compares validation losses: their noise levels and
    noise must be fixed."""
    first = _step(torch.Generator().manual_seed(5))
    torch.randn(100)  # the global stream moves on between validation passes
    second = _step(torch.Generator().manual_seed(5))
    assert torch.equal(first.denoising_loss, second.denoising_loss)
    assert torch.equal(first.noise_level, second.noise_level)
    assert not torch.equal(
        first.denoising_loss, _step(torch.Generator().manual_seed(6)).denoising_loss
    )


def test_the_step_counts_atoms_and_backpropagates_into_the_decoder() -> None:
    decoder = _Decoder()
    result = _step(decoder=decoder)
    assert result.atom_count == 5
    assert result.molecular_descriptor.flat.shape == (2, TARGET_DIMENSION)
    result.denoising_loss.backward()
    assert decoder.linear.weight.grad is not None


def test_every_molecule_draws_its_own_noise_level() -> None:
    decoder = _Decoder()
    result = _step(torch.Generator().manual_seed(0), decoder=decoder)
    assert result.noise_level.shape == (2,)
    assert result.noise_level[0] != result.noise_level[1]
    assert torch.equal(decoder.noise_level, result.noise_level)


def test_the_decoder_input_is_scaled_to_unit_variance() -> None:
    """EDM preconditioning: the decoder sees noised / sqrt(1 + level^2)."""
    level = 3.0
    decoder = _Decoder()
    generator = torch.Generator().manual_seed(0)
    _step(generator, noise=FixedNoiseLevelConfig(level=level), decoder=decoder)
    noised = add_noise(
        _target(),
        PADDING_MASK,
        torch.full((2,), level),
        torch.Generator().manual_seed(0),
    )
    assert decoder.received is not None
    torch.testing.assert_close(
        decoder.received.preprocessed_atomic_embeddings,
        noised / (1.0 + level**2) ** 0.5,
    )


@pytest.mark.parametrize("level", [0.05, 0.5, 5.0])
def test_a_zero_decoder_scores_the_linear_denoiser(level: float) -> None:
    """With the decoder output at zero the estimate is ``noised / (1 + level^2)``.

    The loss must be exactly that estimate's error in the preconditioned units,
    whatever the noise level (1.0 in expectation on unit-variance data).
    """
    result = _step(
        torch.Generator().manual_seed(0),
        noise=FixedNoiseLevelConfig(level=level),
        decoder=_Decoder(zero=True),
    )
    noised = add_noise(
        _target(),
        PADDING_MASK,
        torch.full((2,), level),
        torch.Generator().manual_seed(0),
    )
    estimate = noised / (1.0 + level**2)
    output_scale = level / (1.0 + level**2) ** 0.5
    error = ((estimate - _target()) / output_scale).square()
    expected = error[~PADDING_MASK].mean()
    assert result.denoising_loss.item() == pytest.approx(expected.item(), rel=1e-4)


def test_the_real_decoder_starts_at_the_linear_denoiser_and_gets_no_target_gradient():
    """The assembled encoder / decoder: loss ~1 on unit-variance targets at
    initialisation, and nothing trainable sits in the target's path.

    Regression: the target used to be the output of a trainable projection, so
    the loss could be driven to zero by shrinking the target.
    """
    torch.manual_seed(0)
    atoms, target_dimension = 64, 16
    dimensions = ModelDimensions(
        atomic_descriptor_dimension=target_dimension,
        target_dimension=target_dimension,
        embedding_dimension=32,
        pair_dimension=8,
        radial_basis_dimension=6,
        descriptor_dimension=32,
        descriptor_sequence_length=1,
    )
    attention = AttentionLayerConfig(num_heads=4, dim_feedforward=64, dropout=0.0)
    encoder = EncoderConfig(
        N_layers=2, embedding_dimension=32, attention_layer_config=attention
    ).build(MeanPool(32), dimensions)
    decoder = DecoderConfig(
        N_layers=2, embedding_dimension=24, attention_layer_config=attention
    ).build(dimensions)
    target = torch.randn(4, atoms, target_dimension)
    padding_mask = torch.zeros(4, atoms, dtype=torch.bool)

    class Preprocessor(nn.Module):
        def forward(self, sample: Sample) -> PreprocessedSample:
            return PreprocessedSample(
                preprocessed_atomic_embeddings=target,
                normalized_invariants=target,
                padding_mask=padding_mask,
                initial_pair_representation=torch.zeros(4, atoms, atoms, 8),
                geometrical_encoding=torch.zeros(4, atoms, atoms, 6),
                pair_mask=torch.ones(4, atoms, atoms, dtype=torch.bool),
            )

    target.requires_grad_(True)
    result = denoising_step(
        Sample(atomic_positions=torch.zeros(4 * atoms, 3)),
        preprocessor=Preprocessor(),
        encoder=encoder,
        decoder=decoder,
        noise=LogNormalNoiseLevelConfig(),
    )
    assert result.denoising_loss.item() == pytest.approx(1.0, abs=0.15)
    result.denoising_loss.backward()
    # The only path from the loss to the invariants is the encoder input; with
    # the output head at zero no gradient flows at all, so the target got none.
    assert target.grad is None or torch.count_nonzero(target.grad) == 0
    assert decoder.output_head.weight.grad.abs().sum() > 0


def test_descriptor_statistics_detect_collapse() -> None:
    generator = torch.Generator().manual_seed(0)
    healthy = torch.randn(500, 8, generator=generator)
    collapsed = torch.ones(500, 8)
    one_direction = torch.randn(500, 1, generator=generator) * torch.ones(1, 8)

    assert descriptor_statistics(healthy)["descriptor_effective_rank"] > 7.0
    assert descriptor_statistics(healthy)[
        "descriptor_standard_deviation"
    ] == pytest.approx(1.0, rel=0.1)
    assert descriptor_statistics(collapsed) == {
        "descriptor_standard_deviation": 0.0,
        "descriptor_effective_rank": 0.0,
    }
    assert descriptor_statistics(one_direction)[
        "descriptor_effective_rank"
    ] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="M >= 2"):
        descriptor_statistics(torch.ones(1, 8))
