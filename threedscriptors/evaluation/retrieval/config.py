"""Pydantic configuration for the vector-retrieval eval.

One yaml describes: the dataset, the trained model that produces the
embeddings, the nearest-neighbor index, and a list of retrieval tasks to run
against the resulting :class:`VectorStore`. Tasks are an
``Annotated[..., discriminator="task_kind"]` union so the yaml stays declarative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from threedscriptors.evaluation.benchmark.descriptors import RemediConfig
from threedscriptors.evaluation.retrieval.vector_store import (
    RetrievalIndexConfig,
    SklearnIndexConfig,
)


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


class RetrievalEvalConfig(BaseModel):
    """Full retrieval-eval panel: dataset x model x index x tasks."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: Path
    # Identifier used for the embedding cache filename.
    dataset_id: str
    model: RemediConfig
    output_dir: Path
    index: RetrievalIndexConfig = Field(default_factory=SklearnIndexConfig)
    # Defaults to ``output_dir / "descriptor_cache"`` when omitted.
    descriptor_cache_dir: Path | None = None
    # Optional cap on store size (truncates after embedding) for quick dev runs.
    max_structures: int | None = None
    tasks: list[RetrievalTaskConfig] = Field(default_factory=list, min_length=1)
