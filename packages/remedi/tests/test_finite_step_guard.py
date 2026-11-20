"""FiniteStepGuard: non-finite losses and gradient norms found without a host
sync per step."""

from __future__ import annotations

import pytest
import torch

from remedi.training.pretraining import FiniteStepGuard


def test_finite_steps_pass_every_check() -> None:
    guard = FiniteStepGuard("cpu")
    for step in range(3):
        guard.record(torch.tensor(0.5), torch.tensor(float(step)))
    guard.check(3)
    guard.record(torch.tensor(0.4), torch.tensor(1.0))
    guard.check(4)


@pytest.mark.parametrize(
    ("loss", "gradient_norm"),
    [(float("nan"), 1.0), (1.0, float("inf")), (float("-inf"), float("nan"))],
)
def test_a_non_finite_step_raises_with_the_unchecked_steps(
    loss: float, gradient_norm: float
) -> None:
    guard = FiniteStepGuard("cpu")
    guard.record(torch.tensor(0.5), torch.tensor(1.0))
    guard.check(1)
    guard.record(torch.tensor(loss), torch.tensor(gradient_norm))
    guard.record(torch.tensor(0.5), torch.tensor(1.0))
    with pytest.raises(FloatingPointError, match=r"steps 1\.\.2"):
        guard.check(3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_recording_does_not_synchronise() -> None:
    guard = FiniteStepGuard("cuda")
    loss = torch.tensor(0.5, device="cuda")
    gradient_norm = torch.tensor(1.0, device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        guard.record(loss, gradient_norm)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    guard.check(1)
