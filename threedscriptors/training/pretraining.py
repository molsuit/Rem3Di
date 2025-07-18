import torch

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
    
    masked_squared_diff = squared_diff * ~padding_mask[:,:,None]

    atoms_in_batch = torch.sum((~padding_mask).to(torch.float32))
    
    loss =  1 / atoms_in_batch * torch.sum(masked_squared_diff)# * 1 / noise_level**2
    return loss

