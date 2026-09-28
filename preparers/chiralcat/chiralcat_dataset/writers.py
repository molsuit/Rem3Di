"""Writing the two datasets: extended XYZ geometries plus an index CSV each.

Both CSVs carry full provenance, so no separate audit log is needed:

``dataset.csv``
    index, smiles, label, class_name, n_atoms, source_file, geometry_quality,
    correction, repair_strategy, note
``rejected.csv``
    index, smiles, label, class_name, n_atoms, source_file, stage, reason,
    disposition, detail, has_geometry
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .records import BuildResult, RejectedRecord, Structure

DATASET_COLUMNS = [
    "index",
    "smiles",
    "label",
    "class_name",
    "n_atoms",
    "source_file",
    "geometry_quality",
    "correction",
    "repair_strategy",
    "note",
]

REJECTED_COLUMNS = [
    "index",
    "smiles",
    "label",
    "class_name",
    "n_atoms",
    "source_file",
    "stage",
    "reason",
    "disposition",
    "detail",
    "has_geometry",
]


# Characters an extxyz key=value pair can carry unquoted. Anything else - a
# SMILES bracket, a space, an empty value - has to be quoted or a reader will
# mis-split the comment line.
_BARE_VALUE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.+-"
)


def _comment_line(fields: dict[str, object]) -> str:
    """Build an extxyz comment line, quoting every value that needs it."""
    parts = ["Properties=species:S:1:pos:R:3"]
    for key, value in fields.items():
        text = str(value)
        needs_quotes = not text or not set(text) <= _BARE_VALUE_CHARS
        parts.append(f'{key}="{text}"' if needs_quotes else f"{key}={text}")
    parts.append('pbc="F F F"')
    return " ".join(parts)


def write_dataset_extxyz(structures: list[Structure], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for structure in structures:
            handle.write(f"{structure.n_atoms}\n")
            handle.write(
                _comment_line(
                    {
                        "index": structure.index,
                        "label": structure.label,
                        "class": structure.class_name,
                        "geometry_quality": structure.geometry_quality,
                        "source_file": structure.source_file,
                        "smiles": structure.smiles,
                    }
                )
                + "\n"
            )
            handle.writelines(
                f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
                for symbol, (x, y, z) in zip(
                    structure.symbols, structure.coords, strict=True
                )
            )
    return len(structures)


def write_rejected_extxyz(records: list[RejectedRecord], path: Path) -> int:
    """Write only the rejected records that still carry a usable geometry.

    Most rejections happen precisely because a molecule had no usable geometry,
    so this file is a subset of ``rejected.csv``; the CSV is the complete record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            if not record.has_geometry:
                continue
            handle.write(f"{record.n_atoms}\n")
            handle.write(
                _comment_line(
                    {
                        "index": record.index if record.index is not None else -1,
                        "label": record.label,
                        "class": record.class_name,
                        "stage": record.stage,
                        "reason": record.reason,
                        "disposition": record.disposition,
                        "smiles": record.smiles,
                    }
                )
                + "\n"
            )
            handle.writelines(
                f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
                for symbol, (x, y, z) in zip(record.symbols, record.coords, strict=True)
            )
            written += 1
    return written


def write_dataset_index(structures: list[Structure], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(DATASET_COLUMNS)
        for structure in structures:
            writer.writerow(
                [
                    structure.index,
                    structure.smiles,
                    structure.label,
                    structure.class_name,
                    structure.n_atoms,
                    structure.source_file,
                    structure.geometry_quality,
                    structure.correction,
                    structure.repair_strategy,
                    structure.note,
                ]
            )


def write_rejected_index(records: list[RejectedRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REJECTED_COLUMNS)
        for record in records:
            writer.writerow(
                [
                    "" if record.index is None else record.index,
                    record.smiles,
                    record.label,
                    record.class_name,
                    record.n_atoms,
                    record.source_file,
                    record.stage,
                    record.reason,
                    record.disposition,
                    record.detail,
                    record.has_geometry,
                ]
            )


def write_run_report(result: BuildResult, path: Path, config_dump: dict) -> None:
    """One machine-readable summary of the run: counts, config, audit outcome."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rejected_by_reason: dict[str, int] = {}
    for record in result.rejected:
        key = f"{record.stage}:{record.reason}"
        rejected_by_reason[key] = rejected_by_reason.get(key, 0) + 1

    report = {
        "dataset": {
            "structures": len(result.structures),
            "by_class": result.class_counts,
            "by_geometry_quality": _tally(
                structure.geometry_quality for structure in result.structures
            ),
            "by_source_file": _tally(
                structure.source_file for structure in result.structures
            ),
        },
        "rejected": {
            "total": len(result.rejected),
            "broken": len(result.rejected_broken),
            "filtered": len(result.rejected_filtered),
            "by_stage_and_reason": rejected_by_reason,
        },
        "corrections": {
            "applied": sum(1 for c in result.corrections if c.matched),
            "unmatched": sum(1 for c in result.corrections if not c.matched),
            "records": [
                {
                    "smiles": c.smiles,
                    "action": c.action,
                    "from_class": c.from_class,
                    "to_class": c.to_class,
                    "confidence": c.confidence,
                    "reason": c.reason,
                    "matched": c.matched,
                }
                for c in result.corrections
            ],
        },
        "stereo_audit": {
            "uncovered_mislabels": result.uncovered_mislabels,
        },
        "stage_counts": result.stage_counts,
        "config": config_dump,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _tally(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
