"""Aggregate result model for the vector-retrieval eval.

:class:`RetrievalReport` bundles the per-task results produced over one
:class:`~remedi.evaluation.retrieval.vector_store.VectorStore`. The
store is built and the tasks are run by the framework retrieval task
(:mod:`remedi.evaluation.framework.tasks.retrieval`); this module only
holds the report model they populate.
"""

from __future__ import annotations

from pydantic import BaseModel

from remedi.evaluation.retrieval.nearest_molecule import NearestMoleculeResult
from remedi.evaluation.retrieval.tanimoto_similarity import (
    TanimotoSimilarityResult,
)


class RetrievalReport(BaseModel):
    dataset_id: str
    model_name: str
    n_structures: int
    embedding_dim: int
    tanimoto_results: list[TanimotoSimilarityResult] = []
    nearest_molecule_results: list[NearestMoleculeResult] = []
