import torch
import torch.nn.functional as F

def flat_grad(grads, params):
    """
    grads  : tuple returned by torch.autograd.grad
    params : the parameter list we asked grad() for
    """
    pieces = []
    for p, g in zip(params, grads):
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