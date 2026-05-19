import pytest

from threedscriptors.evaluation.eval001.heads import validate_mlp_device


def test_cpu_mlp_requires_explicit_opt_in():
    with pytest.raises(RuntimeError, match="CPU MLP is blocked"):
        validate_mlp_device("cpu", allow_cpu_mlp=False)


def test_cpu_mlp_opt_in_is_explicit():
    assert validate_mlp_device("cpu", allow_cpu_mlp=True).type == "cpu"
