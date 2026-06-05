"""Losses for supervised single-label classification training.

Focal loss (Lin et al., 2017, *Focal Loss for Dense Object Detection*) is the
objective of choice for the chiral-type task: it down-weights the easy, dominant
classes (achiral / central) so the gradient is not swamped before the model
learns the rare stereochemistry classes (axial / helical / planar). With
``gamma=0`` and uniform ``alpha`` it reduces to (optionally class-weighted)
cross-entropy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FocalLoss(nn.Module):
    r"""Multi-class focal loss.

    ``FL = -alpha_c * (1 - p_t)^gamma * log(p_t)`` where ``p_t`` is the softmax
    probability of the true class ``c`` and ``alpha_c`` an optional per-class
    weight. ``alpha`` is registered as a buffer so it follows the module's
    device / dtype and is restored from a checkpoint.

    Args:
        gamma: focusing parameter; larger -> stronger down-weighting of easy,
            well-classified examples. ``gamma=0`` -> (weighted) cross-entropy.
        alpha: optional per-class weights of shape ``(n_classes,)``; ``None``
            weights every class equally.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: ``(B, n_classes)``; target: ``(B,)`` integer class indices."""
        target = target.view(-1).long()
        log_p = torch.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def inverse_frequency_alpha(
    labels: np.ndarray, n_classes: int, normalize: bool = True
) -> torch.Tensor:
    """Per-class focal ``alpha`` from inverse class frequency on the train set.

    ``alpha_c ∝ N / (n_classes * count_c)`` (the sklearn "balanced" weighting),
    so rare classes are up-weighted. Classes absent from ``labels`` get weight
    1.0. When ``normalize`` the weights are scaled to mean 1 so the loss
    magnitude (and thus a shared learning rate) stays comparable to plain CE.
    """
    labels = np.asarray(labels).reshape(-1).astype(int)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    total = counts.sum()
    weights = np.ones(n_classes, dtype=float)
    nonzero = counts > 0
    weights[nonzero] = total / (n_classes * counts[nonzero])
    if normalize and weights.mean() > 0:
        weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)
