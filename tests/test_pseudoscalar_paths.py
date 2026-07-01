"""Tests for the general 2-TP pseudoscalar path finder and engine.

Expected path sets were enumerated by hand from the selection rules (l-triangle,
parity, l>=1 intermediate, contraction to 0o) and independently cross-checked by a
subagent — see claude/journals/general-pseudo-tp/. Each case asserts the EXACT set of
paths, not just the count.
"""

import warnings

import pytest
import torch
from e3nn.o3 import Irreps

from threedscriptors.model.preprocessing.pseudoscalar_tp import (
    Rem3DiPseudoScalarTP,
    Rem3DiPseudoScalarTPFromVectors,
)
from threedscriptors.utils.pseudoscalar_paths import find_pseudoscalar_paths

# (input irreps, exact expected set of path strings)
PATH_CASES = [
    # same-irrep case: the classic vector triple product, 1 path
    ("1o", {"1o ⊗ 1o → 1e ; 1e ⊗ 1o → 0o"}),
    # proper tensors up to l=2: 3 paths (incl. an l=2 path the old 1o code can't use)
    (
        "0e+1o+2e",
        {
            "1o ⊗ 1o → 1e ; 1e ⊗ 1o → 0o",
            "1o ⊗ 2e → 2o ; 2o ⊗ 2e → 0o",
            "2e ⊗ 2e → 1e ; 1e ⊗ 1o → 0o",
        },
    ),
    # no odd degree available: 0 paths
    ("0e+2e", set()),
    # two odd types: 6 paths
    (
        "1o+2o",
        {
            "1o ⊗ 1o → 1e ; 1e ⊗ 1o → 0o",
            "1o ⊗ 1o → 2e ; 2e ⊗ 2o → 0o",
            "1o ⊗ 2o → 1e ; 1e ⊗ 1o → 0o",
            "1o ⊗ 2o → 2e ; 2e ⊗ 2o → 0o",
            "2o ⊗ 2o → 1e ; 1e ⊗ 1o → 0o",
            "2o ⊗ 2o → 2e ; 2e ⊗ 2o → 0o",
        },
    ),
    # mixed parity: pseudoscalar + axial vector, 1 path
    ("0o+1e", {"0o ⊗ 1e → 1o ; 1o ⊗ 1e → 0o"}),
    # higher l_max: 8 paths
    (
        "1o+2e+3o",
        {
            "1o ⊗ 1o → 1e ; 1e ⊗ 1o → 0o",
            "1o ⊗ 2e → 2o ; 2o ⊗ 2e → 0o",
            "1o ⊗ 3o → 3e ; 3e ⊗ 3o → 0o",
            "2e ⊗ 2e → 1e ; 1e ⊗ 1o → 0o",
            "2e ⊗ 2e → 3e ; 3e ⊗ 3o → 0o",
            "2e ⊗ 3o → 2o ; 2o ⊗ 2e → 0o",
            "3o ⊗ 3o → 1e ; 1e ⊗ 1o → 0o",
            "3o ⊗ 3o → 3e ; 3e ⊗ 3o → 0o",
        },
    ),
]


@pytest.mark.parametrize("irreps_str, expected", PATH_CASES)
def test_find_pseudoscalar_paths(irreps_str, expected):
    found = {str(p) for p in find_pseudoscalar_paths(irreps_str)}
    assert found == expected


@pytest.mark.parametrize("irreps_str, expected", PATH_CASES)
def test_paths_satisfy_selection_rules(irreps_str, expected):
    """Every returned path must obey: l_mid>=1, ir_mid in ir1*ir2, 0o in ir_mid*ir3."""
    for p in find_pseudoscalar_paths(irreps_str):
        assert p.ir_mid.l >= 1
        assert p.ir_mid in list(p.ir_in1 * p.ir_in2)
        outs = list(p.ir_mid * p.ir_in3)
        assert p.ir_out in outs
        assert p.ir_out.l == 0 and p.ir_out.p == -1


def _engine(irreps, K=8, **kw):
    kw.setdefault("invariant_irreps", "16x0e")
    return Rem3DiPseudoScalarTP(irreps, K, **kw).double()


def test_engine_output_shape():
    K = 8
    eng = _engine("0e+4x1o+2x2e", K)
    x = torch.randn(3, eng.input_irreps.dim, dtype=torch.float64)
    inv = torch.randn(3, 16, dtype=torch.float64)
    assert eng(x, inv).shape == (3, K)


def test_engine_inversion_sign_flip():
    """Under spatial inversion the K pseudoscalars must flip sign (non-vacuously)."""
    irreps = Irreps("0e+4x1o+2x2e")  # enough multiplicity to avoid identically-zero
    eng = _engine(irreps)
    x = torch.randn(4, irreps.dim, dtype=torch.float64)
    inv = torch.randn(4, 16, dtype=torch.float64)
    ps = eng(x, inv)
    assert ps.abs().max() > 1e-3  # guard against a vacuous (all-zero) test
    parity = torch.cat(
        [
            torch.full((mul * ir.dim,), float(ir.p), dtype=torch.float64)
            for mul, ir in irreps
        ]
    )
    ps_inv = eng(x * parity, inv)
    assert torch.allclose(ps_inv, -ps, atol=1e-8)


def test_engine_from_vectors_variant():
    K = 8
    fv = Rem3DiPseudoScalarTPFromVectors(
        "0e+4x1o+2x2e", K, invariant_irreps="16x0e"
    ).double()
    inv = torch.randn(2, 16, dtype=torch.float64)
    a = torch.randn(2, fv.factor_a_irreps.dim, dtype=torch.float64)
    b = torch.randn(2, fv.factor_b_irreps.dim, dtype=torch.float64)
    c = torch.randn(2, fv.factor_c_irreps.dim, dtype=torch.float64)
    assert fv((a, b, c), inv).shape == (2, K)


def test_engine_static_weights():
    eng = Rem3DiPseudoScalarTP("0e+4x1o+2x2e", 8, invariant_conditioned=False).double()
    x = torch.randn(2, eng.input_irreps.dim, dtype=torch.float64)
    assert eng(x).shape == (2, 8)


def test_coplanar_warning_fires():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Rem3DiPseudoScalarTP("1x1o", 4, invariant_irreps="16x0e")
    assert any("identically zero" in str(rec.message) for rec in w)


def test_no_paths_raises():
    with pytest.raises(ValueError, match="No 2-TP pseudoscalar paths"):
        Rem3DiPseudoScalarTP("0e+2e", 4, invariant_irreps="16x0e")
