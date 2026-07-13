import torch
import torch.nn.functional as F


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
