"""Enumerate 2-tensor-product paths to a pseudoscalar (``0o``) from a set of irreps.

Theory: a pseudoscalar ``V_0^-`` is reachable from proper/polar tensors only via
**two** tensor products. Valid paths are exactly those where an axial intermediate
``ir_mid`` (l >= 1) contracts with a third input to ``0o``:

  TP1: ir_in1 ⊗ ir_in2 → ir_mid      (triangle rule, parity multiplies)
  TP2: ir_mid ⊗ ir_in3 → 0o          (needs l_mid == l_3 and parity_mid * parity_3 = -1)

The selection engine is e3nn's ``Irrep * Irrep`` (yields CG-allowed outputs obeying
triangle + parity), the same primitive MACE's ``tp_out_irreps_with_instructions`` uses.
"""

from dataclasses import dataclass, field

from e3nn import o3

PSEUDOSCALAR = o3.Irrep(0, -1)  # "0o"


@dataclass(frozen=True)
class PseudoscalarPath:
    """One discovered path ir_in1 ⊗ ir_in2 → ir_mid ; ir_mid ⊗ ir_in3 → 0o."""

    ir_in1: o3.Irrep
    ir_in2: o3.Irrep
    ir_mid: o3.Irrep
    ir_in3: o3.Irrep
    ir_out: o3.Irrep = field(default=PSEUDOSCALAR)

    def __str__(self) -> str:
        return (
            f"{self.ir_in1} ⊗ {self.ir_in2} → {self.ir_mid} ; "
            f"{self.ir_mid} ⊗ {self.ir_in3} → {self.ir_out}"
        )


def _unique_irreps(irreps: o3.Irreps) -> list[o3.Irrep]:
    """Distinct irrep *types* in first-seen order (multiplicities collapsed)."""
    seen: list[o3.Irrep] = []
    for _, ir in irreps:
        if ir not in seen:
            seen.append(ir)
    return seen


def find_pseudoscalar_paths(
    irreps, dedupe_tp1_order: bool = True
) -> list[PseudoscalarPath]:
    """All 2-TP paths from ``irreps`` to a pseudoscalar ``0o``.

    Enumerates over distinct irrep *types* (multiplicity is irrelevant to which paths
    exist). ``dedupe_tp1_order`` treats TP1's two inputs as unordered (ir_a ⊗ ir_b and
    ir_b ⊗ ir_a give the same intermediate set), avoiding duplicate paths. O(D^2 * L).
    """
    irreps = o3.Irreps(irreps)
    types = _unique_irreps(irreps)

    paths: list[PseudoscalarPath] = []
    for i, ir1 in enumerate(types):
        for j, ir2 in enumerate(types):
            if dedupe_tp1_order and j < i:
                continue
            for ir_mid in ir1 * ir2:  # triangle + parity, from e3nn
                if ir_mid.l < 1:  # need an axial (l>=1) intermediate
                    continue
                for ir3 in types:
                    if PSEUDOSCALAR in ir_mid * ir3:
                        paths.append(PseudoscalarPath(ir1, ir2, ir_mid, ir3))
    return paths


def build_stage_instructions(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    target_irreps: o3.Irreps,
    mode: str = "uvu",
) -> tuple[o3.Irreps, list]:
    """Mirror of MACE's ``tp_out_irreps_with_instructions`` with a configurable mode.

    Returns ``(irreps_out, instructions)`` for an ``o3.TensorProduct`` whose outputs are
    exactly those CG paths landing in ``target_irreps``. Output irreps are sorted so the
    following ``o3.Linear`` can simplify them. For non-"uvu"-family modes the output
    multiplicity convention differs, but e3nn validates instruction shapes at build time.
    """
    irreps_in1 = o3.Irreps(irreps_in1)
    irreps_in2 = o3.Irreps(irreps_in2)
    target_irreps = o3.Irreps(target_irreps)

    irreps_out_list: list[tuple[int, o3.Irrep]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_in1):
        for j, (mul2, ir_edge) in enumerate(irreps_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(irreps_out_list)
                    out_mul = mul if mode in ("uvu", "uuu") else mul * mul2
                    irreps_out_list.append((out_mul, ir_out))
                    instructions.append((i, j, k, mode, True))

    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()
    instructions = [
        (i_in1, i_in2, permut[i_out], m, train)
        for i_in1, i_in2, i_out, m, train in instructions
    ]
    instructions = sorted(instructions, key=lambda x: x[2])
    return irreps_out, instructions
