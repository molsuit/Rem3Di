import torch

from threedscriptors.model.pooling import MeanPool


def test_mean_pool_single_real_atom():
    # Single real atom in each batch
    S = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]]
    )  # shape (2, 3, 2)
    mask = torch.tensor(
        [
            [True, True, False],  # only last atom real
            [False, True, True],  # first atom real
        ],
        dtype=torch.bool,
    )  # True -> padded, False -> real

    pool = MeanPool(d_in=2)
    out = pool(S, mask)

    expected = torch.tensor(
        [
            [[5.0, 6.0]],
            [[2.0, 4.0]],
        ]
    )
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"


def test_mean_pool_multiple_real_atoms():
    S = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]],
        ]
    )  # shape (1, 3, 2)
    mask = torch.tensor([[True, False, False]], dtype=torch.bool)

    pool = MeanPool(d_in=2)
    out = pool(S, mask)

    # expected mean = ([3,3] + [5,5]) / 2 = [4,4]; output shape (1, 1, 2).
    expected = torch.tensor([[[4.0, 4.0]]])
    assert torch.allclose(out, expected), f"Expected {expected}, got {out}"


def test_mean_pool_shape_and_dtype():
    B, N, D = 4, 5, 3
    S = torch.randn(B, N, D)
    mask = torch.rand(B, N) > 0.5

    pool = MeanPool(d_in=D)
    out = pool(S, mask)
    # MeanPool now returns a length-1 descriptor sequence (B, 1, D).
    assert out.shape == (B, 1, D), f"Output shape {out.shape} != {(B, 1, D)}"
    assert out.dtype == S.dtype, f"Output dtype {out.dtype} != {S.dtype}"


def test_mean_pool_projects_to_d_out_when_different():
    # Regression: the `agg_mean` ablation run crashed because MeanPool emitted
    # d_in (256) while the decoder cross-attention was wired for output_dim
    # (64). MeanPool must project to d_out, like AttnPool.
    B, N = 2, 5
    pool = MeanPool(d_in=256, d_out=64)
    S = torch.randn(B, N, 256)
    mask = torch.zeros(B, N, dtype=torch.bool)
    out = pool(S, mask)
    assert out.shape == (B, 1, 64)
    assert pool.d_out == 64


def test_mean_pool_identity_when_d_in_equals_d_out():
    pool = MeanPool(d_in=128, d_out=128)
    assert isinstance(pool.out_proj, torch.nn.Identity)
