"""Denoising pretraining: the encoder summarises the clean atomic descriptors
into the molecular descriptor; the decoder gets noised descriptors and the
molecular descriptor and must recover the clean ones.

The target is the standardised MACE invariants (unit variance per channel, no
learnable parameter in their path). Every molecule draws its own noise level,
and the decoder is preconditioned as in EDM (Karras et al. 2022) with a data
deviation of 1, so its regression target has unit variance at every noise
level:

    noised   = target + level * noise
    denoised = skip * noised + output_scale * decoder(input_scale * noised, level)
    skip = 1 / (1 + level^2), output_scale = level / sqrt(1 + level^2),
    input_scale = 1 / sqrt(1 + level^2)

The loss is the mean squared error of the decoder output against
``(target - skip * noised) / output_scale``. A decoder that outputs zero (its
initialisation) scores 1.0: the best a per-channel linear denoiser can do on
unit-variance data. Anything below 1.0 is structure the model has learned.
"""

from dataclasses import dataclass

import torch
from torch import nn

from remedi.configuration.training_config import NoiseLevelConfig
from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.training.data.sample import PreprocessedSample, Sample


def add_noise(
    embeddings: torch.Tensor,
    padding_mask: torch.Tensor,
    noise_level: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Gaussian noise on the real atoms, one standard deviation per molecule.

    ``embeddings`` is ``(B, N, D)``, ``padding_mask`` ``(B, N)`` (True = padding,
    left unchanged), ``noise_level`` ``(B,)``. The result is detached from
    ``embeddings``. ``generator`` (on the embeddings' device) fixes the noise;
    None draws from torch's global generator.
    """
    noise = torch.randn(
        embeddings.shape,
        generator=generator,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    noise = noise.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    return embeddings.detach() + noise_level[:, None, None] * noise


def descriptor_statistics(descriptors: torch.Tensor) -> dict[str, float]:
    """Collapse diagnostics of a ``(M, D)`` set of molecular descriptors.

    ``descriptor_standard_deviation``: mean per-dimension deviation (0 when
    every molecule gets the same descriptor). ``descriptor_effective_rank``:
    participation ratio of the covariance spectrum, between 1 (one direction
    carries all variance) and D.
    """
    if descriptors.dim() != 2 or descriptors.shape[0] < 2:
        raise ValueError(
            "descriptor_statistics needs (M >= 2, D) descriptors, got "
            f"{tuple(descriptors.shape)}"
        )
    descriptors = descriptors.to(torch.float64)
    centred = descriptors - descriptors.mean(dim=0, keepdim=True)
    covariance = centred.T @ centred / (descriptors.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    effective_rank = (
        (total.square() / eigenvalues.square().sum()).item() if total > 0 else 0.0
    )
    return {
        "descriptor_standard_deviation": descriptors.std(dim=0).mean().item(),
        "descriptor_effective_rank": float(effective_rank),
    }


@dataclass(frozen=True)
class DenoisingStepResult:
    denoising_loss: torch.Tensor
    """Mean over real atoms and channels; 1.0 for a decoder that outputs zero."""
    molecular_descriptor: MolecularDescriptor
    atom_count: int
    """Real atoms in the batch."""
    noise_level: torch.Tensor
    """The noise level drawn for each molecule, ``(B,)``."""
    shuffled_descriptor_loss: torch.Tensor | None = None
    """The loss on the same noise with every molecule given its neighbour's
    descriptor (rolled by one in the batch); None unless asked for or with a
    single molecule. Its excess over ``denoising_loss`` is what the descriptor
    contributes."""


def denoising_step(
    samples: Sample,
    *,
    preprocessor: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    noise: NoiseLevelConfig,
    generator: torch.Generator | None = None,
    with_shuffled_descriptor: bool = False,
) -> DenoisingStepResult:
    """One denoising pass, shared by training and validation.

    ``generator`` fixes both the noise levels and the noise. Pair and geometric
    inputs are detached for the decoder. Gradients flow as the caller's grad
    mode allows; none reaches the target. ``with_shuffled_descriptor`` runs the
    decoder a second time with the descriptors rolled by one molecule (a
    diagnostic, for validation).
    """
    if samples.atomic_positions is None:
        raise ValueError("denoising_step needs atomic_positions on the samples")
    # Before the preprocessor: it pads the flat atom arrays in place.
    atom_count = int(samples.atomic_positions.shape[0])
    preprocessed: PreprocessedSample = preprocessor(samples)
    padding_mask = preprocessed.padding_mask
    if preprocessed.normalized_invariants is None or padding_mask is None:
        raise ValueError("the preprocessor returned no normalised invariants or mask")
    if (
        preprocessed.initial_pair_representation is None
        or preprocessed.geometrical_encoding is None
    ):
        raise ValueError("the preprocessor returned no pair or geometric encoding")
    target = preprocessed.normalized_invariants.detach()

    noise_level = noise.sample(
        target.shape[0], device=target.device, generator=generator
    )
    noised = add_noise(target, padding_mask, noise_level, generator)
    level = noise_level[:, None, None]
    input_scale = 1.0 / torch.sqrt(1.0 + level.square())
    skip_scale = 1.0 / (1.0 + level.square())
    output_scale = level / torch.sqrt(1.0 + level.square())

    molecular_descriptor = encoder(preprocessed)
    noised_sample = PreprocessedSample(
        preprocessed_atomic_embeddings=input_scale * noised,
        initial_pair_representation=preprocessed.initial_pair_representation.detach(),
        geometrical_encoding=preprocessed.geometrical_encoding.detach(),
        padding_mask=padding_mask,
        pair_mask=preprocessed.pair_mask,
        atomic_numbers=preprocessed.atomic_numbers,
    )
    decoder_target = (target - skip_scale * noised) / output_scale
    real_atoms = ~padding_mask

    def loss_with(descriptor: MolecularDescriptor) -> torch.Tensor:
        decoder_output = decoder(noised_sample, descriptor, noise_level)
        squared_error = (decoder_output - decoder_target).square()
        squared_error = squared_error * real_atoms[:, :, None]
        return squared_error.sum() / (real_atoms.sum() * target.shape[-1])

    shuffled_descriptor_loss = None
    if with_shuffled_descriptor and molecular_descriptor.batch_size > 1:
        shuffled_descriptor_loss = loss_with(
            MolecularDescriptor(
                tokens=torch.roll(molecular_descriptor.tokens, shifts=1, dims=0)
            )
        )
    return DenoisingStepResult(
        denoising_loss=loss_with(molecular_descriptor),
        molecular_descriptor=molecular_descriptor,
        atom_count=atom_count,
        noise_level=noise_level,
        shuffled_descriptor_loss=shuffled_descriptor_loss,
    )
