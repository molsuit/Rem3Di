import torch.optim as optim
import numpy as np
import torch

from torch.nn.functional import mse_loss
from model.model import Transformer
from torch.optim import Optimizer

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    

def get_random_mask(padding_mask, masking_probability=0.15):
    random_numbers = torch.where(padding_mask.bool(), torch.rand_like(padding_mask), torch.zeros_like(padding_mask))
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


def train_loop(data_loader,model : Transformer,optimizer: Optimizer, hyperparameter,device):

    running_tloss = 0.0
    masking_probability = hyperparameter["masking_probability"]

    model.train()
    optimizer.zero_grad()

    for batch, (embeddings, padding_mask) in enumerate(data_loader):
        
        reconstruction_mask = get_random_mask(padding_mask,masking_probability)

        embeddings = embeddings.to(device)
        padding_mask = padding_mask.to(device)
        reconstruction_mask = reconstruction_mask.to(device)


        # TODO: Harmonize the definition of the padding mask. Torch True = padded, prev: True = not padded

        decoder_prediction = model(embeddings,
                                   padding_mask = torch.logical_not(padding_mask), reconstruction_mask = reconstruction_mask)

        loss = atomic_embedding_loss(decoder_prediction,embeddings,reconstruction_mask)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_tloss += loss.item()

    avg_tloss = running_tloss / (batch + 1)

    return avg_tloss
    

def validation_loop(data_loader,model : Transformer, hyperparameter,device="cuda"):
    
    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
 


    with torch.no_grad():

        model.eval()
        masking_probability = hyperparameter["masking_probability"]


        for batch, (embeddings, padding_mask) in enumerate(data_loader):


            reconstruction_mask = get_random_mask(padding_mask,masking_probability)
            
            embeddings = embeddings.to(device)
            padding_mask = padding_mask.to(device)
            reconstruction_mask = reconstruction_mask.to(device)

            decoder_prediction = model(embeddings,
                                       padding_mask = torch.logical_not(padding_mask), reconstruction_mask = reconstruction_mask)


            loss = atomic_embedding_loss(decoder_prediction,embeddings,reconstruction_mask)
            running_vloss += loss.item()
        avg_vloss = running_vloss / (batch + 1)
        return avg_vloss