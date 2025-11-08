from pathlib import Path

import torch
from mace.calculators.foundations_models import mace_off
from torch_sim.models.mace import MaceModel


def get_torch_sim_mace_off_model(path: Path):

    device = "cuda"
    mace = mace_off(
        model=path,
        default_dtype="float64",
        device=device,
        enable_cueq=True,
        return_raw_model=True,
    )


    mace_model = MaceModel(
        model=mace,
        device=device,
        dtype=torch.float64,
        compute_forces=False,
        compute_stress=False,
        compute_descriptors=True,
        enable_cueq=True,
    )

    return mace_model



def flat_grad(grads, params):
    """
    grads  : tuple returned by torch.autograd.grad
    params : the parameter list we asked grad() for
    """
    pieces = []
    for p, g in zip(params, grads, strict=False):
        if g is None:                 # <- happens when param unused by this task
            g = torch.zeros_like(p)   # treat as 0-gradient
        pieces.append(g.reshape(-1))
    return torch.cat(pieces)

def cosine_matrix(task_losses, shared_params):
    per_task = []
    for loss in task_losses:
        grads = torch.autograd.grad(
            loss, shared_params,
            retain_graph=True,
            allow_unused=True      # keep; we now handle None ourselves
        )
        per_task.append(flat_grad(grads, shared_params).detach())
    G = torch.stack(per_task)       # T × D
    norms = G.norm(dim=1, keepdim=True) + 1e-12
    return (G @ G.T) / (norms @ norms.T)
