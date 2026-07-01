import torch
from e3nn.o3 import Irreps

from threedscriptors.model.preprocessing.pseudoscalar_tp import Rem3DiPseudoScalarTP


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

        # General 2-TP pseudoscalar engine: uses ALL discovered paths to 0o from the
        # equivariant irreps (replaces the hardcoded single vector-triple-product path).
        # TP weights are conditioned on the invariant features.
        self.pseudoscalar_tp = Rem3DiPseudoScalarTP(
            input_irreps=self.equivariant_irreps,
            pseudoscalar_dimension=pseudoscalar_dimension,
            invariant_conditioned=True,
            invariant_irreps=self.invariant_irreps,
            dtype=dtype,
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
        invariant_embeddings: torch.Tensor,
        equivariant_embeddings: torch.Tensor,  # (B, N, F) or (N, F)
        padding: torch.BoolTensor | None = None,  # (B, N), True => padded
    ):
        out = self.pseudoscalar_tp(equivariant_embeddings, invariant_embeddings)

        out = self.ln(out)

        if self.gated:
            out = self.chi_gate(invariant_embeddings) * out

        out = self.linear_out(out)  # (B*N, C) or (N, C)

        # out = self.mlp_out(out)
        out = out.to(torch.float32)
        if padding is not None:
            out = out.masked_fill(padding.unsqueeze(-1), 0.0)
        return out
