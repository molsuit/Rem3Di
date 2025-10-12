import pytest
import torch

# Import your module
# from your_package.pma import PMAAggregator
from threedscriptors.model.pooling import PMAAggregator  


def _make_model(d_in=32, d_out=64, num_heads=4, head_dim=64, k_seeds=3, dropout=0.0, use_mlp=True):
    # head_dim is total Q/K dim; must be divisible by num_heads
    assert head_dim % num_heads == 0
    assert d_out % num_heads == 0
    return PMAAggregator(
        d_in=d_in,
        d_out=d_out,
        num_heads=num_heads,
        head_dim=head_dim,
        k_seeds=k_seeds,
        dropout=dropout,
        use_mlp=use_mlp,
    )


def _finite_tensor(*tensors):
    return all(torch.isfinite(t).all().item() for t in tensors)


@pytest.mark.parametrize("B,N,d_in,d_out,H,head_dim,k_seeds", [
    (2, 11, 32, 64, 4, 64, 3),
    (1,  5, 48, 96, 8, 64, 1),
    (4, 17, 16, 32, 2, 32, 4),
])
@pytest.mark.parametrize("use_mlp", [False, True])
def test_forward_backward_cpu(B,N,d_in,d_out,H,head_dim,k_seeds,use_mlp):
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = _make_model(d_in, d_out, H, head_dim, k_seeds, dropout=0.0, use_mlp=use_mlp).to(device)
    S = torch.randn(B, N, d_in, device=device, dtype=torch.float32, requires_grad=True)
    padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)  # no padding

    out = model(S, padding_mask)
    assert out.shape == (B, d_out)
    assert _finite_tensor(out), "NaN/Inf in forward output"

    loss = (out ** 2).mean()
    loss.backward()

    # Gradients on inputs and parameters should be finite and (mostly) nonzero
    assert S.grad is not None and _finite_tensor(S.grad), "Bad/None grad on inputs"
    # at least some gradient entries should be non-zero
    assert (S.grad.abs() > 0).any().item(), "Zero gradient everywhere on inputs"

    n_param_with_grad = sum(p.grad is not None for p in model.parameters())
    assert n_param_with_grad > 0, "No parameter gradients propagated"
    assert all(_finite_tensor(p.grad) for p in model.parameters() if p.grad is not None), "Param grad has NaN/Inf"


@pytest.mark.parametrize("B,N,d_in,d_out,H,head_dim,k_seeds", [
    (3, 10, 32, 64, 4, 64, 2),
])
def test_with_partial_padding_mask(B,N,d_in,d_out,H,head_dim,k_seeds):
    torch.manual_seed(123)
    device = torch.device("cpu")

    model = _make_model(d_in, d_out, H, head_dim, k_seeds).to(device)
    S = torch.randn(B, N, d_in, device=device, requires_grad=True)

    # Mask last 3 tokens of each sequence
    padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    padding_mask[:, -3:] = True

    out = model(S, padding_mask)
    assert out.shape == (B, d_out)
    assert _finite_tensor(out), "NaN/Inf in forward output with padding"

    loss = out.pow(2).mean()
    loss.backward()
    assert S.grad is not None and _finite_tensor(S.grad)
    assert (S.grad[:, :-3, :].abs().sum() > 0).item(), "Unmasked positions should affect the loss"


@pytest.mark.parametrize("B,N,d_in,d_out,H,head_dim,k_seeds", [
    (2, 7, 24, 48, 4, 64, 3),
])
def test_all_keys_masked_row_is_safe(B,N,d_in,d_out,H,head_dim,k_seeds):
    """
    One batch item has *all* positions masked. This must not produce NaNs.
    For that row, gradients wrt inputs should be (close to) zero.
    """
    torch.manual_seed(7)
    device = torch.device("cpu")

    model = _make_model(d_in, d_out, H, head_dim, k_seeds).to(device)
    S = torch.randn(B, N, d_in, device=device, requires_grad=True)

    padding_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    padding_mask[0, :] = True  # first item fully masked

    out = model(S, padding_mask)
    assert out.shape == (B, d_out)
    assert _finite_tensor(out), "NaN/Inf when an entire row is masked"

    loss = out.pow(2).mean()
    loss.backward()
    assert S.grad is not None and _finite_tensor(S.grad)

    # Gradients for the fully-masked item should be (near) zero
    masked_grad_norm = S.grad[0].norm().item()
    assert masked_grad_norm < 1e-6, f"Expected ~0 grad for fully masked row, got {masked_grad_norm}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward_backward_and_amp():
    torch.manual_seed(42)
    device = torch.device("cuda")

    d_in, d_out, H, head_dim, k_seeds = 32, 64, 4, 64, 3
    model = _make_model(d_in, d_out, H, head_dim, k_seeds, dropout=0.0, use_mlp=True).to(device)
    S = torch.randn(2, 13, d_in, device=device, requires_grad=True)
    padding_mask = torch.zeros(2, 13, dtype=torch.bool, device=device)

    # FP32 path
    out = model(S, padding_mask)
    assert out.shape == (2, d_out)
    assert _finite_tensor(out)
    loss = out.pow(2).mean()
    loss.backward()
    assert S.grad is not None and _finite_tensor(S.grad)

    # AMP path
    S2 = torch.randn(2, 13, d_in, device=device, requires_grad=True)
    padding_mask2 = torch.zeros(2, 13, dtype=torch.bool, device=device)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out2 = model(S2, padding_mask2)
        assert _finite_tensor(out2)
        loss2 = out2.pow(2).mean()

    scaler.scale(loss2).backward()
    scaler.step(optimizer)
    scaler.update()

    # grads should exist and be finite
    assert S2.grad is not None and _finite_tensor(S2.grad)


def test_bad_configuration_raises():
    """
    head_dim must be divisible by num_heads and yield d_k>0;
    d_out must be divisible by num_heads.
    """
    with pytest.raises(AssertionError):
        _ = _make_model(d_in=16, d_out=63, num_heads=4, head_dim=64, k_seeds=2)  # d_out not divisible
    with pytest.raises(AssertionError):
        _ = _make_model(d_in=16, d_out=64, num_heads=8, head_dim=30, k_seeds=2)  # 30%8!=0
