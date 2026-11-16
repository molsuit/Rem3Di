from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from remedi.model.molecular_descriptor import MolecularDescriptor
from remedi.training.data.sample import PreprocessedSample, Sample


def get_random_mask(padding_mask, masking_probability=0.15):
    # Padding mask denotes the padded atoms that should not be masked during pretraining
    random_numbers = torch.where(
        torch.logical_not(padding_mask.bool()),
        torch.rand_like(padding_mask),
        torch.ones_like(padding_mask),
    )
    reconstruction_mask = random_numbers < masking_probability
    return reconstruction_mask


def atomic_embedding_loss(decoder_prediction, atomic_embedding, reconstruction_mask):
    """
    decoder_prediction: torch.Tensor of shape (batch_size, set_dim, embedding_dim)
    atomic_embedding: torch.Tensor of shape (batch_size, set_dim, embedding_dim)
    reconstruction_mask: torch.Tensor of shape (batch_size, set_dim)
    """

    mask = reconstruction_mask.unsqueeze(-1).type_as(decoder_prediction)

    # Compute squared differences
    squared_diff = (decoder_prediction - atomic_embedding) ** 2

    # Apply the mask
    masked_squared_diff = squared_diff * mask

    # Compute the mean loss over the masked elements
    loss = masked_squared_diff.sum() / mask.sum()
    return loss


def atom_denoising_loss(
    input_atomic_embeddings, denoised_embeddings, padding_mask, noise_level
):
    # Compute squared differences
    squared_diff = (denoised_embeddings - input_atomic_embeddings) ** 2
    # Apply the mask

    masked_squared_diff = squared_diff * ~padding_mask[:, :, None]

    atoms_in_batch = torch.sum((~padding_mask).to(torch.float32))

    loss = 1 / atoms_in_batch * torch.sum(masked_squared_diff) * 1 / noise_level**2
    return loss


def vicreg_descriptor_loss(
    descriptor: torch.Tensor,
    *,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance regularizers on a (B, D) descriptor batch.

    Returns the *unweighted* (variance_loss, covariance_loss); the caller
    applies weights so each component can be logged separately.
    """
    if descriptor.dim() != 2:
        raise ValueError(
            f"vicreg_descriptor_loss expects a (B, D) tensor, got {descriptor.shape}"
        )
    if descriptor.shape[0] < 2:
        zero = descriptor.new_zeros(())
        return zero, zero

    std = descriptor.std(dim=0) + eps
    variance_loss = F.relu(target_std - std).mean()

    centered = descriptor - descriptor.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (descriptor.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    covariance_loss = off_diag.pow(2).sum() / descriptor.shape[1]

    return variance_loss, covariance_loss


def add_noise(
    embeddings: torch.Tensor,
    padding_mask: torch.Tensor,
    noise_level: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Gaussian noise of standard deviation ``noise_level`` on the real atoms.

    ``embeddings`` is ``(B, N, D)``, ``padding_mask`` ``(B, N)`` (True = padding,
    left unchanged). The result is detached from ``embeddings``. ``generator``
    (on the embeddings' device) fixes the noise; None draws from torch's global
    generator.
    """
    noise = torch.randn(
        embeddings.shape,
        generator=generator,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    noise = noise.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    return embeddings.detach() + noise_level * noise


@dataclass(frozen=True)
class DenoisingStepResult:
    denoising_loss: torch.Tensor
    molecular_descriptor: MolecularDescriptor
    atom_count: int
    """Real atoms in the batch."""


def denoising_step(
    samples: Sample,
    *,
    preprocessor: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    noise_level: float,
    generator: torch.Generator | None = None,
) -> DenoisingStepResult:
    """One denoising pass, shared by training and validation.

    The encoder sees the clean atomic embeddings and summarises them into the
    molecular descriptor; the decoder gets the noised embeddings and the
    descriptor and must recover the clean ones. Pair and geometric inputs are
    detached for the decoder. Gradients flow as the caller's grad mode allows.
    """
    if samples.atomic_positions is None:
        raise ValueError("denoising_step needs atomic_positions on the samples")
    # Before the preprocessor: it pads the flat atom arrays in place.
    atom_count = int(samples.atomic_positions.shape[0])
    preprocessed: PreprocessedSample = preprocessor(samples)
    embeddings = preprocessed.preprocessed_atomic_embeddings
    padding_mask = preprocessed.padding_mask
    if embeddings is None or padding_mask is None:
        raise ValueError("the preprocessor returned no atomic embeddings or mask")
    if (
        preprocessed.initial_pair_representation is None
        or preprocessed.geometrical_encoding is None
    ):
        raise ValueError("the preprocessor returned no pair or geometric encoding")
    clean_embeddings = embeddings.clone()
    noised_embeddings = add_noise(embeddings, padding_mask, noise_level, generator)

    molecular_descriptor = encoder(preprocessed)
    noised_sample = PreprocessedSample(
        preprocessed_atomic_embeddings=noised_embeddings,
        initial_pair_representation=preprocessed.initial_pair_representation.detach(),
        geometrical_encoding=preprocessed.geometrical_encoding.detach(),
        padding_mask=padding_mask,
        pair_mask=preprocessed.pair_mask,
    )
    denoised_embeddings = decoder(noised_sample, molecular_descriptor)
    denoising_loss = atom_denoising_loss(
        clean_embeddings,
        denoised_embeddings,
        padding_mask=padding_mask,
        noise_level=noise_level,
    )
    return DenoisingStepResult(
        denoising_loss=denoising_loss,
        molecular_descriptor=molecular_descriptor,
        atom_count=atom_count,
    )
