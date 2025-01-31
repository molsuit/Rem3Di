import torch
from model.model import Transformer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_random_mask(padding_mask, masking_probability=0.15):
    random_numbers = torch.where(
        padding_mask.bool(),
        torch.rand_like(padding_mask),
        torch.zeros_like(padding_mask),
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


def train_loop(
    data_loader,
    model: Transformer,
    optimizer: Optimizer,
    scheduler,
    hyperparameter,
    device,
):
    running_tloss = 0.0
    masking_probability = hyperparameter["masking_probability"]

    model.train()
    optimizer.zero_grad()

    for batch, (embeddings, padding_mask) in enumerate(data_loader):
        reconstruction_mask = get_random_mask(padding_mask, masking_probability)

        embeddings = embeddings.to(device)
        padding_mask = padding_mask.to(device)
        reconstruction_mask = reconstruction_mask.to(device)

        # TODO: Harmonize the definition of the padding mask. Torch True = padded, prev: True = not padded

        decoder_prediction = model(
            embeddings,
            padding_mask=torch.logical_not(padding_mask),
            reconstruction_mask=reconstruction_mask,
        )

        loss = atomic_embedding_loss(
            decoder_prediction, embeddings, reconstruction_mask
        )

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_tloss += loss.item()

    avg_tloss = running_tloss / (batch + 1)

    return avg_tloss


def validation_loop(data_loader, model: Transformer, hyperparameter, device="cuda"):
    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.

    with torch.no_grad():
        model.eval()
        masking_probability = hyperparameter["masking_probability"]

        for batch, (embeddings, padding_mask) in enumerate(data_loader):
            reconstruction_mask = get_random_mask(padding_mask, masking_probability)

            embeddings = embeddings.to(device)
            padding_mask = padding_mask.to(device)
            reconstruction_mask = reconstruction_mask.to(device)

            decoder_prediction = model(
                embeddings,
                padding_mask=torch.logical_not(padding_mask),
                reconstruction_mask=reconstruction_mask,
            )

            loss = atomic_embedding_loss(
                decoder_prediction, embeddings, reconstruction_mask
            )
            running_vloss += loss.item()
        avg_vloss = running_vloss / (batch + 1)
        return avg_vloss


def get_warmup_inverse_sqrt_scheduler(optimizer, warmup_steps, d_model):
    def lr_lambda(current_step):
        if current_step == 0:
            return 0
        scale = d_model**-0.5
        if current_step < warmup_steps:
            return scale * (current_step * warmup_steps**-1.5)
        return scale * (current_step**-0.5)

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)
