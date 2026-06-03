"""Retrieval-eval runner: build one vector store, run the configured tasks.

Mirrors ``benchmark/runner.py``: open the dataset, embed it into a
:class:`VectorStore` (cached), dispatch each task, and serialize results to
``output_dir`` (per-task yaml + a combined report; a CSV for the Tanimoto rows).
"""

from __future__ import annotations

import logging

import pandas as pd
import pydantic_yaml as pyd_yaml
from pydantic import BaseModel

from threedscriptors.data_handling.dataset.molecule_dataset import MoleculeDataset
from threedscriptors.evaluation.retrieval.config import (
    NearestMoleculeTaskConfig,
    RetrievalEvalConfig,
    TanimotoSimilarityTaskConfig,
)
from threedscriptors.evaluation.retrieval.nearest_molecule import (
    NearestMoleculeResult,
    run_nearest_molecule,
)
from threedscriptors.evaluation.retrieval.tanimoto_similarity import (
    TanimotoSimilarityResult,
    run_tanimoto_similarity,
)
from threedscriptors.evaluation.retrieval.vector_store import build_vector_store

logger = logging.getLogger(__name__)


class RetrievalReport(BaseModel):
    dataset_id: str
    model_name: str
    n_structures: int
    embedding_dim: int
    tanimoto_results: list[TanimotoSimilarityResult] = []
    nearest_molecule_results: list[NearestMoleculeResult] = []


def run_retrieval_eval(cfg: RetrievalEvalConfig) -> RetrievalReport:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.descriptor_cache_dir or (cfg.output_dir / "descriptor_cache")

    dataset = MoleculeDataset.open_existing_dataset_from_dir(cfg.dataset_path)
    logger.info(
        "Building vector store for %s (%d structures) with model %s",
        cfg.dataset_id,
        dataset.N_structures,
        cfg.model.name,
    )
    store = build_vector_store(
        cfg.model,
        dataset,
        cache_dir=cache_dir,
        index_config=cfg.index,
        dataset_id=cfg.dataset_id,
        max_structures=cfg.max_structures,
    )

    report = RetrievalReport(
        dataset_id=cfg.dataset_id,
        model_name=cfg.model.name,
        n_structures=store.n,
        embedding_dim=store.dim,
    )

    for task_cfg in cfg.tasks:
        if isinstance(task_cfg, TanimotoSimilarityTaskConfig):
            res = run_tanimoto_similarity(store, task_cfg, cfg.output_dir)
            report.tanimoto_results.append(res)
            pyd_yaml.to_yaml_file(cfg.output_dir / f"{task_cfg.name}.yaml", res)
        elif isinstance(task_cfg, NearestMoleculeTaskConfig):
            res = run_nearest_molecule(store, task_cfg, cfg.output_dir)
            report.nearest_molecule_results.append(res)
            pyd_yaml.to_yaml_file(cfg.output_dir / f"{task_cfg.name}.yaml", res)
        else:  # pragma: no cover - exhaustive by the discriminated union
            raise ValueError(f"Unknown retrieval task config: {task_cfg!r}")

    pyd_yaml.to_yaml_file(cfg.output_dir / "retrieval_report.yaml", report)
    if report.tanimoto_results:
        pd.DataFrame(
            [r.model_dump() for r in report.tanimoto_results]
        ).to_csv(cfg.output_dir / "tanimoto_results.csv", index=False)

    logger.info("Retrieval eval complete: results written to %s", cfg.output_dir)
    return report
