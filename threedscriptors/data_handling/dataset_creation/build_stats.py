"""Runtime accounting surfaced by the dataset-build orchestrator.

Generators populate :class:`LoadStats` while iterating their source (raw vs
invalid SMILES vs filter rejects vs canonical-SMILES duplicates). Pipeline
stages populate :class:`StageStats` for failures that occur after the
generator (e.g. RDKit ETKDG / MMFF embedding crashes inside
``ConformerGenerationStage``). The orchestrator collects whichever fields
are present and logs them next to the existing per-stage timings at finalize.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LoadStats:
    n_raw_rows: int = 0
    n_invalid_smiles: int = 0
    n_filtered_out: int = 0
    n_duplicates: int = 0
    n_kept: int = 0

    def summary(self) -> str:
        return (
            f"kept {self.n_kept}/{self.n_raw_rows} "
            f"(dropped: {self.n_invalid_smiles} invalid, "
            f"{self.n_filtered_out} filtered, "
            f"{self.n_duplicates} duplicate)"
        )


@dataclass(slots=True)
class StageStats:
    n_attempted: int = 0
    n_value_errors: int = 0
    n_other_errors: int = 0
    n_emitted: int = 0

    @property
    def n_failed(self) -> int:
        return self.n_value_errors + self.n_other_errors

    def summary(self) -> str:
        return (
            f"emitted {self.n_emitted}/{self.n_attempted} "
            f"(failed: {self.n_value_errors} value-error, "
            f"{self.n_other_errors} other)"
        )
