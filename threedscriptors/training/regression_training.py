import torch
from torch import nn

from threedscriptors.data_handling.sample import Sample
from threedscriptors.model.model_output import ModelOutput


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


def chiral_difference_loss(predictions, labels, regression_mask):

    # predictions is stacked along the batch dimension, enantiomer 1 at the top, enantiomer2 at the bottom. We only want to penalize the difference of the predicted retention time.

    B = predictions.shape[0]
    if B % 2 != 0:
        raise ValueError(f"Batch size must be even — got {B}")
    N = B // 2

    pred_e1, pred_e2 = predictions[:N], predictions[N:]
    lab_e1,  lab_e2  = labels[:N],        labels[N:]

    print(predictions)
    print(labels)

    mask_e1 = regression_mask[:N].bool()
    mask_e2 = regression_mask[N:].bool()

    valid = mask_e1 & mask_e2

    diff_pred = pred_e1[valid] - pred_e2[valid]
    diff_lab  = lab_e1[valid]  - lab_e2[valid]

    real_loss = (predictions-labels).pow(2).mean()

    loss = (diff_pred - diff_lab).pow(2).mean()

    return loss

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



class ChiralDifferenceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, sample: Sample, output : ModelOutput):

        return
