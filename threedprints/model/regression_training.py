import torch


def multitask_masked_loss(predictions, labels, regression_mask):
    # Calculate the elementwise squared error
    squared_error = (predictions - labels)**2
    masked_squared_error = squared_error * regression_mask

    active_labels_count = regression_mask.sum(dim=1) # Weigh the loss by the number of active labels
    active_labels_count = torch.where(active_labels_count == 0, torch.ones_like(active_labels_count), active_labels_count) # Avoid division by zero. For tasks with no label in the batch, the masked squared error in that column will be zero anyways.
    weighted_loss = (masked_squared_error.sum(dim=1) / active_labels_count)
    final_loss = weighted_loss.mean() # TODO: Is this the correct weighing or should the error just be the sum, because we previously already weighed with the number of labels in each task?

    # might actually be good to also return the individual contributions of the loss for tracking purposes
    return final_loss
