import torch

from threedscriptors.model.pooling import AttnPool, MeanPool




def test_mean_pool_single_real_atom():
    # Single real atom in each batch
    S = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]
    ])  # shape (2, 3, 2)
    mask = torch.tensor([
        [True, True, False],   # only last atom real
        [False, True, True]    # first atom real
    ], dtype=torch.bool)      # True -> padded, False -> real

    pool = MeanPool()
    out = pool(S, mask)

    # expected means
    expected = torch.tensor([
        [5.0, 6.0],   # from [5.0, 6.0]
        [2.0, 4.0]    # from [2.0, 4.0]
    ])
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"


def test_mean_pool_multiple_real_atoms():
    # Multiple real atoms
    S = torch.tensor([
        [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
    ])  # shape (1, 3, 2)
    mask = torch.tensor([
        [True, False, False]
    ], dtype=torch.bool)  # ignore first, keep next two

    pool = MeanPool()
    out = pool(S, mask)

    # expected mean = ([3,3] + [5,5]) / 2 = [4,4]
    expected = torch.tensor([[4.0, 4.0]])
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"


def test_mean_pool_shape_and_dtype():
    # Check that output shape and dtype match expected
    B, N, D = 4, 5, 3
    S = torch.randn(B, N, D)
    mask = torch.rand(B, N) > 0.5

    pool = MeanPool()
    out = pool(S, mask)
    assert out.shape == (B, D), f"Output shape {out.shape} != {(B, D)}"
    assert out.dtype == S.dtype, f"Output dtype {out.dtype} != {S.dtype}"
