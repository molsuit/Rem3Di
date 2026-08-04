"""Pre-rewrite chiral embedding, kept so published checkpoints stay loadable.

:class:`~remedi.model.preprocessing.chiral_embedding_model.ChiralEmbeddingModel`
now builds its pseudoscalars with :class:`Rem3DiPseudoScalarTP`, which discovers
every path to ``0o`` from the equivariant irreps and conditions the tensor-product
weights on the invariant features. The module it replaced used a single hardcoded
vector-triple-product path with statically learned TP weights.

That is a change of architecture, not of naming, so checkpoints do not transfer:

===========================  ========================================
pre-rewrite                  current
===========================  ========================================
``lin0``, ``lin1``, ``lin2`` ``pseudoscalar_tp.lin_a/lin_b/lin_c``
                             (same shapes, so these alone would map)
``tp_cross.weight`` (4096,)  ``pseudoscalar_tp.tp1.weight`` is ``(0,)``
``tp_dot.weight``   (4096,)  ``pseudoscalar_tp.tp2.weight`` is ``(0,)``
—                            ``pseudoscalar_tp.lin_mid``
—                            ``pseudoscalar_tp.weight_mlp1/weight_mlp2``
===========================  ========================================

The old static weights have no counterpart: they were learned tensors, and the
current module generates the equivalent quantities from an MLP. There is no
mapping from one to the other, which is why this is preserved verbatim rather
than migrated.

Selected by ``legacy_pseudoscalar: true`` on the embedding preprocess config. New
work should use the current module.
"""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.o3 import Irreps

from remedi.model.preprocessing.chiral_embedding_model import ChiGate


class ChiralEmbeddingModelLegacy(torch.nn.Module):
    """Pseudoscalar features via one vector triple product.

    Drop-in for :class:`ChiralEmbeddingModel` — identical constructor signature
    and identical output contract — differing only in how the pseudoscalar is
    formed.

    Three ``o3.Linear`` maps take the equivariant features to ``Kx1o``. A cross
    product gives ``Kx1e``, and a dot with the third vector gives ``Kx0o``: a
    genuine pseudoscalar, odd under inversion, so it separates enantiomers. The
    result is layer-normed, optionally gated by an invariant-conditioned sigmoid,
    and projected to the chiral embedding width.
    """

    def __init__(
        self,
        invariant_irreps: Irreps,
        equivariant_irreps: Irreps,
        pseudoscalar_dimension: int,
        chiral_embedding_dim: int,
        gated: bool = True,
        dtype=torch.float64,
    ) -> None:
        super().__init__()

        self.dtype = dtype
        self.gated = gated
        self.invariant_irreps = invariant_irreps
        self.equivariant_irreps = equivariant_irreps

        self.pseudoscalar_irreps = Irreps(f"{pseudoscalar_dimension}x0o")
        self.eq_embedding_irrep = Irreps(f"{pseudoscalar_dimension}x1o")

        self.lin0 = o3.Linear(self.equivariant_irreps, self.eq_embedding_irrep)
        self.lin1 = o3.Linear(self.equivariant_irreps, self.eq_embedding_irrep)
        self.lin2 = o3.Linear(self.equivariant_irreps, self.eq_embedding_irrep)

        # (1o x 1o) -> 1e, the cross product. Single CG path, weights are the CG
        # coefficients themselves.
        self.tp_cross = o3.TensorProduct(
            self.eq_embedding_irrep,
            self.eq_embedding_irrep,
            o3.Irreps(f"{pseudoscalar_dimension}x1e"),
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        # (1e x 1o) -> 0o, the dot. Odd parity, hence chirality-sensitive.
        self.tp_dot = o3.TensorProduct(
            self.tp_cross.irreps_out,
            self.eq_embedding_irrep,
            self.pseudoscalar_irreps,
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        self.ln = torch.nn.LayerNorm(pseudoscalar_dimension, dtype=dtype, bias=False)

        self.chi_gate = ChiGate(
            inv_dim=self.invariant_irreps.dim, K=self.pseudoscalar_irreps.dim
        )

        self.linear_out = torch.nn.Linear(
            pseudoscalar_dimension, chiral_embedding_dim, bias=False
        )

    def forward(
        self,
        invariant_embeddings: torch.Tensor,
        equivariant_embeddings: torch.Tensor,
        padding: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        x0 = self.lin0(equivariant_embeddings)
        x1 = self.lin1(equivariant_embeddings)
        x2 = self.lin2(equivariant_embeddings)

        cross = self.tp_cross(x0, x1)
        out = self.tp_dot(cross, x2)

        out = self.ln(out)

        if self.gated:
            out = self.chi_gate(invariant_embeddings) * out

        out = self.linear_out(out)
        out = out.to(torch.float32)

        if padding is not None:
            out = out.masked_fill(padding.unsqueeze(-1), 0.0)
        return out
