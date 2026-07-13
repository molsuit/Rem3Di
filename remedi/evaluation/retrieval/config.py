"""Pydantic configuration for the vector-retrieval tasks.

Defines the retrieval sub-task configs and their
``Annotated[..., discriminator="task_kind"]`` union, consumed by the framework
retrieval task (:mod:`remedi.evaluation.framework.tasks.retrieval`),
which supplies the dataset / model / index at the manifest level.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class TanimotoSimilarityTaskConfig(BaseModel):
    """Task A: does embedding proximity recover fingerprint (Tanimoto) similarity?"""

    task_kind: Literal["tanimoto_similarity"] = "tanimoto_similarity"
    name: str = "tanimoto_similarity"
    # Neighbors per query used for the within-neighborhood distribution + overlap.
    k: int = 10
    # Number of query molecules sampled to form the within-kNN distribution and
    # the overlap@k average (each sampled query is compared against all N).
    n_query_sample: int = 500
    # Random molecule pairs drawn to estimate the global Tanimoto baseline.
    n_global_pairs: int = 50_000
    # Morgan fingerprint parameters.
    fp_radius: int = 2
    fp_n_bits: int = 2048
    seed: int = 0


class NearestMoleculeTaskConfig(BaseModel):
    """Task C: retrieve the closest dataset molecules to each query."""

    task_kind: Literal["nearest_molecule"] = "nearest_molecule"
    name: str = "nearest_molecule"
    k: int = 10
    # Query by brand-new molecules (embedded through the model via a generated
    # 3D conformer) and/or by the row index of an existing store entry.
    query_smiles: list[str] = Field(default_factory=list)
    query_indices: list[int] = Field(default_factory=list)
    # Additionally sample this many SMILES uniformly at random from the dataset
    # and use them as (re-embedded) queries. A useful round-trip check: a fresh
    # conformer of a known molecule should retrieve itself / close analogues.
    n_random_query_smiles: int = 0
    query_sample_seed: int = 0
    # ETKDG seed for conformer generation of SMILES queries.
    conformer_seed: int = 0xF00D
    # How many queries to render as a "query + top-k neighbors" molecule grid
    # (PNG per query). 0 disables the visualization. Capped at the number of
    # embedded queries that actually returned neighbors.
    n_visualize: int = 0
    # Molecules per row in each grid; defaults to k + 1 (query + all neighbors
    # on one row) when None.
    viz_mols_per_row: int | None = None

    def has_queries(self) -> bool:
        return (
            bool(self.query_smiles)
            or bool(self.query_indices)
            or self.n_random_query_smiles > 0
        )


RetrievalTaskConfig = Annotated[
    TanimotoSimilarityTaskConfig | NearestMoleculeTaskConfig,
    Field(discriminator="task_kind"),
]
