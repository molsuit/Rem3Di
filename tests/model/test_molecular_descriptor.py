import pytest
import torch

from threedscriptors.model.molecular_descriptor import MolecularDescriptor


def test_basic_shapes_and_flat():
    tokens = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    desc = MolecularDescriptor(tokens=tokens)

    assert desc.batch_size == 2
    assert desc.seq_len == 3
    assert desc.dim == 4
    assert desc.flat_dim == 12
    assert desc.flat.shape == (2, 12)
    # Flatten preserves row-major ordering of (L, D).
    assert torch.equal(desc.flat[0], tokens[0].reshape(-1))


def test_rejects_non_3d_tensor():
    with pytest.raises(ValueError):
        MolecularDescriptor(tokens=torch.zeros(2, 4))
    with pytest.raises(ValueError):
        MolecularDescriptor(tokens=torch.zeros(2, 3, 4, 1))


def test_to_and_detach_return_new_wrapper():
    tokens = torch.randn(1, 2, 3, requires_grad=True)
    desc = MolecularDescriptor(tokens=tokens)

    detached = desc.detach()
    assert detached.tokens.requires_grad is False
    assert desc.tokens.requires_grad is True
    assert torch.equal(detached.tokens, desc.tokens.detach())

    moved = desc.to("cpu")
    assert moved.tokens.device == torch.device("cpu")


def test_indexing_returns_descriptor():
    tokens = torch.randn(4, 2, 3)
    desc = MolecularDescriptor(tokens=tokens)

    sliced = desc[1:3]
    assert isinstance(sliced, MolecularDescriptor)
    assert sliced.tokens.shape == (2, 2, 3)

    # Single-batch indexing rewraps to (1, L, D).
    one = desc[0]
    assert one.tokens.shape == (1, 2, 3)
