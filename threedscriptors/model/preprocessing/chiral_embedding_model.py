
import torch
from e3nn import o3
from e3nn.o3 import Irreps


class ChiGate(torch.nn.Module):
    """
    FiLM-style invariant→gate mapping.
    Returns a (…, K) tensor in (0, 1).
    """

    def __init__(self, inv_dim: int, K: int, hidden: int | None = None):
        super().__init__()
        if hidden is None:
            hidden = 2 * inv_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inv_dim, hidden, bias=None),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, K, bias=None),
        )
        self.act = torch.nn.Sigmoid()  # keep outputs positive

    def forward(self, inv):
        gate = self.act(self.net(inv))  # (..., K)
        return gate


class OddMLP(torch.nn.Module):

    def __init__(
        self, pseudoscalar_dim: int, hidden_dim: int, chiral_embedding_dim: int
    ):

        super().__init__()

        self.pseudoscalar_dim = pseudoscalar_dim
        self.hidden_dim = hidden_dim
        self.chiral_embedding_dim = chiral_embedding_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(pseudoscalar_dim, hidden_dim, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, chiral_embedding_dim, bias=False),
        )

    def forward(self, x):
        return self.model(x)


class ChiralEmbeddingModel(torch.nn.Module):

    def __init__(
        self,
        invariant_irreps: Irreps,
        equivariant_irreps: Irreps,
        pseudoscalar_dimension: Irreps,
        chiral_embedding_dim: int,
        gated: bool = True,
        dtype=torch.float64,
    ):
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

        # self.eq_norm = NormActivation(self.eq_embedding_irrep, torch.sigmoid )

        self.tp_cross = o3.TensorProduct(
            self.eq_embedding_irrep,
            self.eq_embedding_irrep,
            o3.Irreps(f"{pseudoscalar_dimension}x1e"),
            # single CG path, weights = CG only
            instructions=[(0, 0, 0, "uvu", True)],
            internal_weights=True,
            shared_weights=True,
            irrep_normalization="component",
        )

        # 2) dot: (1e ⊗ 1o) → 0o
        self.tp_dot = o3.TensorProduct(
            self.tp_cross.irreps_out,
            self.eq_embedding_irrep,
            self.pseudoscalar_irreps,  # final pseudoscalar
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

        # self.mlp_out = OddMLP(
        #    pseudoscalar_dim = pseudoscalar_dimension,
        #    hidden_dim = pseudoscalar_dimension * 2,
        #    chiral_embedding_dim = chiral_embedding_dim
        # )

    def forward(
        self,
        invariant_embeddings: torch.Tensor, equivariant_embeddings: torch.Tensor,   # (B, N, F) or (N, F)
        padding: torch.BoolTensor | None = None,  # (B, N), True => padded
    ):



        x0, x1, x2 = self.lin0(equivariant_embeddings), self.lin1(equivariant_embeddings), self.lin2(equivariant_embeddings)

        cross = self.tp_cross(x0, x1)
        out = self.tp_dot(cross, x2)

        out = self.ln(out)

        if self.gated:
            out = self.chi_gate(invariant_embeddings) * out

        out = self.linear_out(out).to(torch.float32)  # (B*N, C) or (N, C)

        if padding is not None:
            out = out.masked_fill(padding.unsqueeze(-1), 0.0)
        return out
