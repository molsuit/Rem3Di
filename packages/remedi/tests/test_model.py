import pytest
import torch

from remedi.model.preprocessing.atomic_descriptor_preprocessor import (
    PrecomputedInvariantNormalization,
)


def test_set_stats_refuses_overwrite_by_default():
    norm = PrecomputedInvariantNormalization(invariant_dimension=256)

    mean = torch.ones(256)
    std = torch.ones(256)

    norm.set_stats(mean, std)

    with pytest.raises(RuntimeError):
        # Reregistering stats should fail without overwrite=True to prevent
        # silently changing normalization after the model has been trained.
        norm.set_stats(mean, std)


def test_set_stats_overwrite_allowed_when_requested():
    norm = PrecomputedInvariantNormalization(invariant_dimension=256)

    norm.set_stats(torch.ones(256), torch.ones(256))
    norm.set_stats(2 * torch.ones(256), 3 * torch.ones(256), overwrite=True)

    assert torch.all(norm.mean == 2.0)
    assert torch.all(norm.std == 3.0)


def test_stats_roundtrip_through_state_dict(tmp_path):
    norm = PrecomputedInvariantNormalization(invariant_dimension=256)

    mean = 2 * torch.ones(256)
    std = 3 * torch.ones(256)

    norm.set_stats(mean, std)

    file = tmp_path / "norm.pth"
    torch.save(norm.state_dict(), file)

    reloaded = PrecomputedInvariantNormalization(invariant_dimension=256)
    reloaded.load_state_dict(torch.load(file))

    assert torch.all(reloaded.mean == mean.view(1, 1, -1))
    assert torch.all(reloaded.std == std.view(1, 1, -1))
