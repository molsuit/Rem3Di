"""Retrieval task: Tanimoto-similarity + nearest-molecule over one embedding.

Pulls the embedding matrix from a shared :class:`EmbeddingSpec` and the fitted
index from a shared :class:`IndexSpec` (rather than embedding + indexing inline),
so the two retrieval sub-tasks — and any descriptor-analysis task on the same
dataset — reuse one embedding and one index per run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.benchmark.descriptors import RemediCalculator
from threedscriptors.evaluation.framework.context import EvalContext
from threedscriptors.evaluation.framework.resources import EmbeddingSpec, IndexSpec
from threedscriptors.evaluation.results import EvalResult, PydanticResult, TableResult
from threedscriptors.evaluation.retrieval.config import (
    NearestMoleculeTaskConfig,
    TanimotoSimilarityTaskConfig,
)
from threedscriptors.evaluation.retrieval.config import (
    RetrievalTaskConfig as RetrievalSubTaskConfig,
)
from threedscriptors.evaluation.retrieval.nearest_molecule import (
    nearest_molecule_figures,
    run_nearest_molecule,
)
from threedscriptors.evaluation.retrieval.runner import RetrievalReport
from threedscriptors.evaluation.retrieval.tanimoto_similarity import (
    run_tanimoto_similarity,
)
from threedscriptors.evaluation.retrieval.vector_store import (
    RetrievalIndexConfig,
    SklearnIndexConfig,
    VectorStore,
)

logger = logging.getLogger(__name__)


class RetrievalConfig(BaseModel):
    """Vector-retrieval panel over one dataset embedded by the run's model."""

    kind: Literal["retrieval"] = "retrieval"
    dataset_path: Path
    dataset_id: str
    index: RetrievalIndexConfig = Field(default_factory=SklearnIndexConfig)
    max_structures: int | None = None
    tasks: list[RetrievalSubTaskConfig] = Field(default_factory=list, min_length=1)

    def run(self, ctx: EvalContext) -> Iterator[EvalResult]:
        out = ctx.task_dir("retrieval")
        store = self._build_store(ctx)
        report = RetrievalReport(
            dataset_id=self.dataset_id,
            model_name=ctx.model.name,
            n_structures=store.n,
            embedding_dim=store.dim,
        )

        for task_cfg in self.tasks:
            if isinstance(task_cfg, TanimotoSimilarityTaskConfig):
                res = run_tanimoto_similarity(store, task_cfg, out)
                report.tanimoto_results.append(res)
            elif isinstance(task_cfg, NearestMoleculeTaskConfig):
                res = run_nearest_molecule(store, task_cfg, out)
                report.nearest_molecule_results.append(res)
            else:  # pragma: no cover - exhaustive by the discriminated union
                raise ValueError(f"Unknown retrieval task config: {task_cfg!r}")
            yield PydanticResult(
                file_name=Path(f"retrieval/{task_cfg.name}.yaml"), obj=res
            )
            if isinstance(task_cfg, NearestMoleculeTaskConfig):
                yield from nearest_molecule_figures(store, res, task_cfg)

        yield PydanticResult(
            file_name=Path("retrieval/retrieval_report.yaml"), obj=report
        )
        if report.tanimoto_results:
            yield TableResult(
                file_name=Path("retrieval/tanimoto_results.csv"),
                frame=pd.DataFrame([r.model_dump() for r in report.tanimoto_results]),
            )

    def _build_store(self, ctx: EvalContext) -> VectorStore:
        dataset = MoleculeDataset.open_existing_dataset_from_dir(self.dataset_path)
        emb_spec = EmbeddingSpec(
            dataset_id=self.dataset_id,
            descriptor=ctx.model,
            dataset=dataset,
            cache_dir=ctx.resource_cache_dir,
        )
        X_full = ctx.resources.get(emb_spec)
        n = X_full.shape[0] if self.max_structures is None else min(
            self.max_structures, X_full.shape[0]
        )
        X = np.ascontiguousarray(X_full[:n], dtype=np.float32)

        if self.max_structures is None:
            index = ctx.resources.get(IndexSpec(emb_spec, self.index))
        else:
            # A capped smoke run indexes only the prefix, so it cannot share the
            # full-dataset index resource.
            index = self.index.build()
            index.fit(X)

        # Build the run's model as an embedder so the nearest-molecule task can
        # re-embed fresh query conformers. Only the structure-based REM3DI
        # calculator can; an ECFP-style model leaves embedder None and that task
        # raises a clear error if requested.
        calc = ctx.model.build()
        embedder = calc if isinstance(calc, RemediCalculator) else None

        return VectorStore(
            X=X,
            smiles=dataset.get_smiles_per_structure()[:n],
            structure_ids=np.asarray(dataset.structure_ids[:n]),
            index=index,
            embedder=embedder,
        )
