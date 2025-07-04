import torch
from torch import nn
from threedscriptors.model.model_output import ModelOutput
from threedscriptors.data_handling.sample import Sample


def multitask_masked_loss(predictions, labels, regression_mask):
    # Calculate the elementwise squared error
    squared_error = (predictions - labels) ** 2


    masked_squared_error = squared_error * regression_mask
    active_labels_count = regression_mask.sum(
        dim=0
    )  # Weigh the loss by the number of active labels
    active_labels_count = torch.where(
        active_labels_count == 0,
        torch.ones_like(active_labels_count),
        active_labels_count,
    )

    # Avoid division by zero. For tasks with no label in the batch, the masked squared error in that column will be zero anyways.
    weighted_loss = masked_squared_error.sum(dim=0) / active_labels_count

    return weighted_loss

class BaseMultitaskLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, sample: Sample, output : ModelOutput):
        
        loss_per_task = multitask_masked_loss(predictions= output.regression_predictions, labels= sample.regression_targets, regression_mask= sample.regression_masks)

        loss = loss_per_task.mean()

        return loss, loss_per_task

class DynamicallyWeighedMultitaskLoss(nn.Module):

    def __init__(self, N_tasks):

        super().__init__()

        self.log_vars = nn.Parameter(torch.zeros(N_tasks))

    def forward(self, sample: Sample, output : ModelOutput):

        loss_per_task = multitask_masked_loss(predictions= output.regression_predictions, labels= sample.regression_targets, regression_mask= sample.regression_masks)

        precision = torch.exp(-self.log_vars)

        loss = (precision * loss_per_task + self.log_vars).sum()

        return loss, loss_per_task  


