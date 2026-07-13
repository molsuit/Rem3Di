"""Per-molecule timing records for ConformerGenerationStage.

Each ``ConformerTimingRecord`` captures the wall-time spent in ETKDG embedding
and MMFF relaxation for one molecule, plus the outcome status. Records are
collected by ``ConformerGenerationStage`` across all batches and serialized to
``<output_root>/<dataset_id>/conformer_timings.jsonl`` at orchestrator
finalize, so the per-molecule distribution survives the build as an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

ConformerTimingStatus = Literal[
    "ok",
    "embed_failed",
    "mmff_failed",
    "value_error",
    "other_error",
]


class ConformerTimingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    isomeric_smiles: str
    n_atoms: int
    n_confs_requested: int
    n_confs_emitted: int
    t_embed_s: float
    t_mmff_s: float
    status: ConformerTimingStatus
    error_msg: str | None = None


@dataclass
class EmbedResult:
    """Worker return type for ``embed_one_smiles`` — always populated, never raises."""

    nonisomeric_smiles: str | None
    positions: np.ndarray | None
    atomic_numbers: np.ndarray | None
    timing: ConformerTimingRecord


def write_timings_jsonl(records: list[ConformerTimingRecord], path: Path) -> None:
    """Truncate-write the timing records as JSONL (one record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
