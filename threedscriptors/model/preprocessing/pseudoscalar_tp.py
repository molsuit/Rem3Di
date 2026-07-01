"""General 2-tensor-product pseudoscalar block built from arbitrary input irreps.

``Rem3DiPseudoScalarTP`` replaces the hardcoded single vector-triple-product path in
``ChiralEmbeddingModel`` with *all* discovered 2-TP paths to ``0o`` (see
``threedscriptors.utils.pseudoscalar_paths``), wrapped as two fused ``o3.TensorProduct``s
modeled on MACE's interaction block. It returns K raw pseudoscalars; LayerNorm / gating /
projection stay in ``ChiralEmbeddingModel``.
"""

import warnings
from collections import Counter

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps

from threedscriptors.utils.pseudoscalar_paths import (
    PSEUDOSCALAR,
    build_stage_instructions,
    find_pseudoscalar_paths,
)

_UVU_FAMILY = ("uvu", "uuu")


class Rem3DiPseudoScalarTP(torch.nn.Module):
    """K pseudoscalars from a single input tensor via two tensor products.

    Discovers every path ``ir1 ⊗ ir2 → ir_mid ; ir_mid ⊗ ir3 → 0o`` valid for
    ``input_irreps`` and realizes them as two fused TPs. Convention (per plan): an
    ``o3.Linear`` sits on every TP *input* and none on the outputs — ``lin_a``/``lin_b``
    feed TP1, the mandatory ``lin_mid`` feeds TP1's output into TP2, ``lin_c`` feeds TP2's
    second input. Optional invariant-conditioned TP weights (on by default).
    """

    def __init__(
        self,
        input_irreps,
        pseudoscalar_dimension: int,
        connection_mode: str = "uvu",
        invariant_conditioned: bool = True,
        invariant_irreps=None,
        weight_hidden_dim: int = 16,
        add_output_linear: bool = False,
        dtype: torch.dtype = torch.float32,
        _from_vectors: bool = False,
    ):
        super().__init__()

        self.input_irreps = Irreps(input_irreps)
        self.K = pseudoscalar_dimension
        self.mode = connection_mode
        self.invariant_conditioned = invariant_conditioned
        self.add_output_linear = add_output_linear
        self.from_vectors = _from_vectors
        self.pseudoscalar_irreps = Irreps(f"{self.K}x0o")

        self.paths = find_pseudoscalar_paths(self.input_irreps)
        if not self.paths:
            raise ValueError(
                f"No 2-TP pseudoscalar paths for input irreps {self.input_irreps}. "
                "Need at least one odd-sum (l1+l2+l3) path — e.g. an l>=1 axial "
                "intermediate that contracts with a third input to 0o."
            )

        # Factor irreps: K copies of each degree appearing as a 1st/2nd/3rd TP input.
        self.factor_a_irreps = self._factor_irreps(p.ir_in1 for p in self.paths)
        self.factor_b_irreps = self._factor_irreps(p.ir_in2 for p in self.paths)
        self.factor_c_irreps = self._factor_irreps(p.ir_in3 for p in self.paths)

        # Axial intermediates (TP1 target / TP2 first input).
        mids = _unique([p.ir_mid for p in self.paths])
        mid_target = Irreps([(1, ir) for ir in mids])
        self.irreps_mid = Irreps([(self.K, ir) for ir in mids])

        self._warn_coplanar()

        # Input linears (skipped in the from-vectors variant).
        if self.from_vectors:
            self.lin_a = self.lin_b = self.lin_c = None
        else:
            self.lin_a = o3.Linear(self.input_irreps, self.factor_a_irreps)
            self.lin_b = o3.Linear(self.input_irreps, self.factor_b_irreps)
            self.lin_c = o3.Linear(self.input_irreps, self.factor_c_irreps)

        # TP1: factor_a ⊗ factor_b → axial intermediates.
        irreps_mid_raw, instr1 = build_stage_instructions(
            self.factor_a_irreps, self.factor_b_irreps, mid_target, self.mode
        )
        self.tp1 = o3.TensorProduct(
            self.factor_a_irreps,
            self.factor_b_irreps,
            irreps_mid_raw,
            instructions=instr1,
            internal_weights=not invariant_conditioned,
            shared_weights=not invariant_conditioned,
            irrep_normalization="component",
        )

        # Mandatory linear on the intermediate (mixes TP1 paths feeding the same degree).
        self.lin_mid = o3.Linear(irreps_mid_raw, self.irreps_mid)

        # TP2: intermediate ⊗ factor_c → K x 0o. All matching CG paths accumulate into
        # the single output block (uvu-style), so the output is exactly K x 0o.
        instr2 = self._contract_to_pseudoscalar_instructions()
        self.tp2 = o3.TensorProduct(
            self.irreps_mid,
            self.factor_c_irreps,
            self.pseudoscalar_irreps,
            instructions=instr2,
            internal_weights=not invariant_conditioned,
            shared_weights=not invariant_conditioned,
            irrep_normalization="component",
        )

        self.lin_out = (
            o3.Linear(self.pseudoscalar_irreps, self.pseudoscalar_irreps)
            if add_output_linear
            else None
        )

        # Invariant-conditioned TP weights.
        if invariant_conditioned:
            if invariant_irreps is None:
                raise ValueError(
                    "invariant_conditioned=True requires invariant_irreps to size the "
                    "weight-generating MLPs."
                )
            inv_dim = Irreps(invariant_irreps).dim
            self.weight_mlp1 = FullyConnectedNet(
                [inv_dim, weight_hidden_dim, self.tp1.weight_numel], F.silu
            )
            self.weight_mlp2 = FullyConnectedNet(
                [inv_dim, weight_hidden_dim, self.tp2.weight_numel], F.silu
            )
        else:
            self.weight_mlp1 = self.weight_mlp2 = None

        self.to(dtype)

    def _factor_irreps(self, irs) -> Irreps:
        return Irreps([(self.K, ir) for ir in _unique(list(irs))])

    def _contract_to_pseudoscalar_instructions(self) -> list:
        instr = []
        for i, (_, ir_i) in enumerate(self.irreps_mid):
            for j, (_, ir_j) in enumerate(self.factor_c_irreps):
                if PSEUDOSCALAR in ir_i * ir_j:
                    instr.append((i, j, 0, self.mode, True))
        return instr

    def _warn_coplanar(self) -> None:
        input_mult: dict = {}
        for mul, ir in self.input_irreps:
            input_mult[ir] = input_mult.get(ir, 0) + mul
        for path in self.paths:
            need = Counter([path.ir_in1, path.ir_in2, path.ir_in3])
            for ir, count in need.items():
                have = input_mult.get(ir, 0)
                if have < count:
                    warnings.warn(
                        f"Path '{path}' reuses irrep {ir} {count}x across its factors "
                        f"but input multiplicity is {have} (< {count}); the mixed "
                        "factors cannot be linearly independent, so this pseudoscalar "
                        "may be identically zero.",
                        stacklevel=2,
                    )

    def _project_inputs(self, x):
        if self.from_vectors:
            a, b, c = x
            return a, b, c
        return self.lin_a(x), self.lin_b(x), self.lin_c(x)

    def forward(self, x, invariants=None):
        a, b, c = self._project_inputs(x)

        if self.invariant_conditioned:
            if invariants is None:
                raise ValueError("invariant_conditioned=True requires `invariants`.")
            mid_raw = self.tp1(a, b, self.weight_mlp1(invariants))
            mid = self.lin_mid(mid_raw)
            ps = self.tp2(mid, c, self.weight_mlp2(invariants))
        else:
            mid = self.lin_mid(self.tp1(a, b))
            ps = self.tp2(mid, c)

        if self.lin_out is not None:
            ps = self.lin_out(ps)
        return ps  # [..., K]


class Rem3DiPseudoScalarTPFromVectors(Rem3DiPseudoScalarTP):
    """From-vectors variant: ``forward((a, b, c), invariants=None)`` consumes factors
    already projected to ``factor_a/b/c_irreps`` (no input linears created)."""

    def __init__(self, *args, **kwargs):
        kwargs["_from_vectors"] = True
        super().__init__(*args, **kwargs)


def _unique(items: list) -> list:
    out: list = []
    for it in items:
        if it not in out:
            out.append(it)
    return out
