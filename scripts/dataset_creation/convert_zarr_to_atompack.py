"""Convert datasets from the zarr layout to ``structures.atp`` beside ``table.parquet`` (one-off).

``BENCHMARK_DATA_FORMAT.md`` §3.3. For each dataset directory:

1. the table's ``structure_id``, ``molecule_id``, ``stereoisomer_id``,
   ``total_charge`` and ``multiplicity`` must equal the zarr's, and the zarr must
   still match the ``structures_sha256`` its ``provenance.yaml`` records (the
   zarr-era definition), so nothing is converted from damaged arrays;
2. the zarr arrays are read directly with ``zarr`` and written to
   ``structures.atp`` through ``write_structures``, streamed in pieces;
3. ``structures.atp`` is read back and every array (pointer, atomic numbers,
   positions) compared exactly with the zarr's;
4. ``structures_sha256`` (the §3.3 definition) is re-recorded in
   ``provenance.yaml`` and in every ``selections/<name>.yaml`` that recorded the
   zarr's hash (the frames are proven the same, only the definition moved),
   nothing else in them changes; the table is not rewritten, so its hashes stay;
5. only after every check passed, the zarr goes (``positions``,
   ``atomic_numbers``, ``molecule_ptr``, ``total_charge``, ``multiplicity``,
   ``ids``, ``zarr.json``, ``dataset_config.yaml``), unless ``--keep-zarr``;
6. ``verify_dataset`` runs on the result.

A dataset that already has ``structures.atp`` and no zarr is reported
``unchanged`` (and still verified), so the script is safe to rerun. A run that
stopped after writing ``structures.atp`` is finished from where it stopped:
an existing ``structures.atp`` is compared with the zarr, never rewritten.

``--dry-run`` converts a shadow of each dataset under ``--shadow-root``
(``<shadow-root>/<dataset name>``: symbolic links to the source's entries,
copies of the files the conversion rewrites, ``selections/*.yaml`` included)
and leaves the source untouched;
the shadow stays, so it can be evaluated. Deleting the zarr in a shadow removes
the links only.

Usage (a path is a dataset directory or a directory holding datasets)::

    uv run python scripts/dataset_creation/convert_zarr_to_atompack.py --dry-run \\
        --shadow-root /tmp/shadow benchmark_data/datasets
    uv run python scripts/dataset_creation/convert_zarr_to_atompack.py \\
        --report conversion_report.yaml benchmark_data/datasets
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import zarr
from pydantic import BaseModel, ConfigDict, Field
from zarr.storage import LocalStore

from remedi_data.dataset import StructureFile, Structures, write_structures
from remedi_data.dataset.dataset import structures_sha256
from remedi_data.format import (
    PROVENANCE_FILENAME,
    SPEC_FILENAME,
    STRUCTURES_FILENAME,
    BundleProvenance,
    read_provenance,
    read_table,
)
from remedi_data.format.verify import verify_dataset
from remedi_data.logging_setup import setup_logging
from remedi_data.parallel import default_workers
from remedi_data.selection.select import SELECTIONS_DIRECTORY
from remedi_data.yaml_io import read_yaml, write_model, write_yaml

logger = logging.getLogger("convert_zarr_to_atompack")

#: Every entry of the zarr layout, deleted once the conversion is checked;
#: ``zarr.json`` first, so a deletion that stops halfway leaves no readable zarr
#: behind, only leftovers a rerun removes.
ZARR_ENTRIES = (
    "zarr.json",
    "positions",
    "atomic_numbers",
    "molecule_ptr",
    "total_charge",
    "multiplicity",
    "ids",
    "dataset_config.yaml",
)
#: The id arrays of the zarr and the table columns they must equal.
ID_ARRAYS = ("structure_id", "molecule_id", "stereoisomer_id")
#: Files the conversion rewrites; a shadow holds copies of them, not links.
REWRITTEN_FILES = (PROVENANCE_FILENAME,)
#: Structures per piece when streaming, comparing and hashing.
PIECE_STRUCTURES = 200_000

Outcome = Literal["unchanged", "converted", "would_convert", "failed"]


class DatasetConversionReport(BaseModel):
    """What converting one dataset did, or would do on a dry run."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: Path
    #: Where the conversion happened: the dataset itself, or its shadow.
    converted_path: Path | None = None
    outcome: Outcome
    changes: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    structures: int = 0
    atoms: int = 0
    recorded_structures_sha256: str | None = None
    structures_sha256: str | None = None
    #: Bytes and files of the zarr entries and of ``structures.atp``.
    zarr_bytes: int = 0
    zarr_files: int = 0
    structures_file_bytes: int = 0
    write_seconds: float = 0.0
    compare_seconds: float = 0.0
    verify_seconds: float = 0.0
    elapsed_seconds: float = 0.0


class ConversionReport(BaseModel):
    """Every dataset of one run."""

    model_config = ConfigDict(extra="forbid")

    dry_run: bool
    keep_zarr: bool
    datasets: list[DatasetConversionReport] = Field(default_factory=list)

    def count(self, outcome: Outcome) -> int:
        return sum(dataset.outcome == outcome for dataset in self.datasets)


class ConversionError(RuntimeError):
    """A dataset that cannot be converted as it is; its zarr was not touched."""


def has_zarr(directory: Path) -> bool:
    return (directory / "zarr.json").is_file()


def is_convertible_directory(directory: Path) -> bool:
    """A dataset in either layout: ``dataset.yaml`` plus a zarr or ``structures.atp``."""
    return (directory / SPEC_FILENAME).is_file() and (
        has_zarr(directory) or (directory / STRUCTURES_FILENAME).is_file()
    )


def dataset_directories(paths: list[Path]) -> list[Path]:
    """The dataset directories the given datasets and roots of datasets stand for.

    Raises:
        ConversionError: if a path is neither a dataset nor holds any.
    """
    datasets: list[Path] = []
    for path in map(Path, paths):
        if is_convertible_directory(path):
            datasets.append(path)
            continue
        children = (
            sorted(child for child in path.iterdir() if is_convertible_directory(child))
            if path.is_dir()
            else []
        )
        if not children:
            raise ConversionError(f"{path} is not a dataset and holds no datasets")
        datasets += children
    return datasets


def _array(group: zarr.Group, name: str) -> zarr.Array:
    node = group[name]
    if not isinstance(node, zarr.Array):
        raise ConversionError(f"the zarr's {name} is not an array")
    return node


class ZarrDataset:
    """The arrays of one zarr-layout dataset, read directly with ``zarr``."""

    def __init__(self, directory: Path) -> None:
        group = zarr.open_group(store=LocalStore(directory, read_only=True), mode="r")
        self.positions = _array(group, "positions")
        self.atomic_numbers = _array(group, "atomic_numbers")
        self.pointer = np.asarray(_array(group, "molecule_ptr")[:], dtype=np.int64)
        self.total_charge = np.asarray(_array(group, "total_charge")[:])
        self.multiplicity = np.asarray(_array(group, "multiplicity")[:])
        self.ids = {
            name: np.asarray(_array(group, f"ids/{name}")[:]) for name in ID_ARRAYS
        }
        for name, array, expected in (
            ("positions", self.positions, np.dtype(np.float32)),
            ("atomic_numbers", self.atomic_numbers, np.dtype(np.uint8)),
        ):
            if np.dtype(array.dtype) != expected:
                raise ConversionError(
                    f"the zarr's {name} is {array.dtype}, not {expected}"
                )

    def __len__(self) -> int:
        return len(self.pointer) - 1

    def piece(self, start: int, stop: int) -> Structures:
        """The frames of structures ``start:stop``, read from the zarr."""
        first_atom, last_atom = int(self.pointer[start]), int(self.pointer[stop])
        return Structures.from_arrays(
            self.positions[first_atom:last_atom],
            self.atomic_numbers[first_atom:last_atom],
            self.pointer[start : stop + 1] - first_atom,
        )

    def pieces(self) -> Iterator[Structures]:
        for start in range(0, len(self), PIECE_STRUCTURES):
            yield self.piece(start, min(start + PIECE_STRUCTURES, len(self)))

    def zarr_era_sha256(self) -> str:
        """``structures_sha256`` as the zarr era defined it (each array with a
        ``name:dtype:shape`` header, in this order), streamed in pieces."""
        digest = hashlib.sha256()
        for name, array in (
            ("atomic_numbers", self.atomic_numbers),
            ("positions", self.positions),
        ):
            digest.update(
                f"{name}:{np.dtype(array.dtype).str}:{array.shape}\n".encode()
            )
            atoms = array.shape[0]
            step = PIECE_STRUCTURES * 64
            for start in range(0, atoms, step):
                digest.update(np.ascontiguousarray(array[start : start + step]))
        for name, values in (
            ("molecule_ptr", self.pointer),
            ("total_charge", self.total_charge),
            ("multiplicity", self.multiplicity),
        ):
            digest.update(f"{name}:{values.dtype.str}:{values.shape}\n".encode())
            digest.update(np.ascontiguousarray(values))
        return digest.hexdigest()


def table_problems(directory: Path, zarr_dataset: ZarrDataset) -> list[str]:
    """Where the table's ids, charges and multiplicities differ from the zarr's."""
    table = read_table(directory, columns=[*ID_ARRAYS, "total_charge", "multiplicity"])
    if len(table) != len(zarr_dataset):
        return [
            f"the zarr holds {len(zarr_dataset)} structures, the table {len(table)}"
        ]
    problems = [
        f"the table's {name} differs from the zarr's ids/{name}"
        for name in ID_ARRAYS
        if not np.array_equal(
            table[name].to_numpy(dtype=np.int64),
            zarr_dataset.ids[name].astype(np.int64),
        )
    ]
    for name, values in (
        ("total_charge", zarr_dataset.total_charge),
        ("multiplicity", zarr_dataset.multiplicity),
    ):
        if not np.array_equal(
            table[name].to_numpy(dtype=np.float64), values.astype(np.float64)
        ):
            problems.append(f"the table's {name} differs from the zarr's {name}")
    return problems


def comparison_problems(
    structure_file: StructureFile, zarr_dataset: ZarrDataset
) -> list[str]:
    """Every array of ``structures.atp`` against the zarr's, exactly, piece by piece."""
    if len(structure_file) != len(zarr_dataset):
        return [
            f"{STRUCTURES_FILENAME} holds {len(structure_file)} structures, "
            f"the zarr {len(zarr_dataset)}"
        ]
    problems: list[str] = []
    for start in range(0, len(zarr_dataset), PIECE_STRUCTURES):
        stop = min(start + PIECE_STRUCTURES, len(zarr_dataset))
        expected = zarr_dataset.piece(start, stop)
        written = structure_file.slice(start, stop)
        for name in ("ptr", "atomic_numbers", "positions"):
            left, right = getattr(written, name), getattr(expected, name)
            if left.dtype != right.dtype or not np.array_equal(left, right):
                problems.append(
                    f"{STRUCTURES_FILENAME} {name} of structures {start}:{stop} "
                    "differs from the zarr"
                )
    return problems


def _zarr_size(directory: Path) -> tuple[int, int]:
    """Bytes and files of the zarr entries of ``directory``."""
    total_bytes = total_files = 0
    for name in ZARR_ENTRIES:
        if (directory / name).exists():
            entry_bytes, entry_files = _directory_size(directory / name)
            total_bytes += entry_bytes
            total_files += entry_files
    return total_bytes, total_files


def _present_zarr_entries(directory: Path) -> list[Path]:
    return [
        directory / name
        for name in ZARR_ENTRIES
        if (directory / name).exists() or (directory / name).is_symlink()
    ]


def _recorded_structures_sha256(directory: Path) -> str | None:
    outputs = read_provenance(directory).outputs
    return None if outputs is None else outputs.structures_sha256


def _current_structures_sha256(directory: Path) -> str:
    """``structures_sha256`` (§3.3) of ``structures.atp`` and the table's charges."""
    table = read_table(directory, columns=["total_charge", "multiplicity"])
    return structures_sha256(
        StructureFile.open(directory / STRUCTURES_FILENAME),
        table["total_charge"].to_numpy(dtype=np.float64),
        table["multiplicity"].to_numpy(dtype=np.float64),
    )


def _directory_size(path: Path) -> tuple[int, int]:
    """Bytes and files under ``path`` (a file counts itself); links are followed
    for a shadow's entries, so a shadow reports the source's sizes."""
    if path.is_file():
        return path.stat().st_size, 1
    sizes = [entry.stat().st_size for entry in path.rglob("*") if entry.is_file()]
    return sum(sizes), len(sizes)


def _remove(path: Path) -> None:
    """Delete one zarr entry; in a shadow, the link only."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _replace_yaml(path: Path, document: dict) -> None:
    """Write ``document`` to ``path`` under a temporary name and rename it.

    Raises:
        ConversionError: if ``path`` is a link (a shadow must never write
            through to its source).
    """
    if path.is_symlink():
        raise ConversionError(f"{path} is a link; refusing to write through it")
    staging = path.with_name(f".{path.name}.partial")
    write_yaml(staging, document)
    staging.replace(path)


def _rewrite_selection_hashes(directory: Path, old: str, new: str) -> list[str]:
    """Set ``structures_sha256`` from ``old`` to ``new`` in every selection record
    of ``directory`` that holds ``old``; returns their relative paths."""
    rewritten: list[str] = []
    for path in sorted((directory / SELECTIONS_DIRECTORY).glob("*.yaml")):
        document = read_yaml(path)
        if not isinstance(document, dict) or document.get("structures_sha256") != old:
            continue
        document["structures_sha256"] = new
        _replace_yaml(path, document)
        rewritten.append(f"{SELECTIONS_DIRECTORY}/{path.name}")
    return rewritten


def _rewrite_structures_hash(directory: Path, value: str) -> None:
    """Set ``outputs.structures_sha256`` in ``provenance.yaml``, nothing else.

    The document is edited as loaded and dumped with the writer's own options,
    so every other line stays as it was; it is written under a temporary name
    and renamed.
    """
    path = directory / PROVENANCE_FILENAME
    document = read_yaml(path)
    document["outputs"]["structures_sha256"] = value
    BundleProvenance.model_validate(document)
    _replace_yaml(path, document)


def _check_the_zarr(
    directory: Path,
    zarr_dataset: ZarrDataset,
    zarr_era_hash: str,
    report: DatasetConversionReport,
) -> None:
    """The table against the zarr, and the zarr against its recorded hash.

    Either hash will do: the zarr-era one of the zarr, or (an earlier run
    stopped after re-recording it) the current one of an existing
    ``structures.atp``, which is then compared with the zarr.

    Raises:
        ConversionError: listing every problem.
    """
    problems = table_problems(directory, zarr_dataset)
    report.recorded_structures_sha256 = _recorded_structures_sha256(directory)
    if zarr_era_hash != report.recorded_structures_sha256 and (
        not (directory / STRUCTURES_FILENAME).exists()
        or _current_structures_sha256(directory) != report.recorded_structures_sha256
    ):
        problems.append(
            "the zarr does not match the structures_sha256 its provenance.yaml records"
        )
    if problems:
        raise ConversionError("; ".join(problems))


def _write_and_compare(
    directory: Path, zarr_dataset: ZarrDataset, report: DatasetConversionReport
) -> None:
    """Write ``structures.atp`` unless it exists, then compare it with the zarr.

    Raises:
        ConversionError: if they differ; a file written by this call is removed.
    """
    structures_path = directory / STRUCTURES_FILENAME
    wrote_now = not structures_path.exists()
    if wrote_now:
        started = time.monotonic()
        write_structures(structures_path, zarr_dataset.pieces())
        report.write_seconds = time.monotonic() - started
        report.changes.append(f"wrote {STRUCTURES_FILENAME}")
    started = time.monotonic()
    problems = comparison_problems(StructureFile.open(structures_path), zarr_dataset)
    report.compare_seconds = time.monotonic() - started
    if problems:
        if wrote_now:
            structures_path.unlink()
        raise ConversionError("; ".join(problems))
    report.structures_file_bytes = structures_path.stat().st_size


def convert_layout(
    directory: Path, report: DatasetConversionReport, *, keep_zarr: bool
) -> None:
    """Convert ``directory`` in place and record what happened in ``report``.

    Raises:
        ConversionError: if a check fails; the zarr is untouched then (a
            ``structures.atp`` written by this call is removed again).
    """
    if not has_zarr(directory):
        _remove_leftovers(directory, report)
        return
    zarr_dataset = ZarrDataset(directory)
    zarr_era_hash = zarr_dataset.zarr_era_sha256()
    report.structures = len(zarr_dataset)
    report.atoms = int(zarr_dataset.pointer[-1])
    report.zarr_bytes, report.zarr_files = _zarr_size(directory)
    _check_the_zarr(directory, zarr_dataset, zarr_era_hash, report)
    _write_and_compare(directory, zarr_dataset, report)

    report.structures_sha256 = _current_structures_sha256(directory)
    if report.structures_sha256 != report.recorded_structures_sha256:
        _rewrite_structures_hash(directory, report.structures_sha256)
        report.changes.append(
            "provenance.yaml: structures_sha256 "
            f"{report.recorded_structures_sha256} -> {report.structures_sha256}"
        )
    for path in _rewrite_selection_hashes(
        directory, zarr_era_hash, report.structures_sha256
    ):
        report.changes.append(f"{path}: structures_sha256 re-recorded")
    if keep_zarr:
        return
    for path in _present_zarr_entries(directory):
        _remove(path)
    report.changes.append(f"deleted the zarr ({', '.join(ZARR_ENTRIES)})")


def _remove_leftovers(directory: Path, report: DatasetConversionReport) -> None:
    """Finish a conversion whose zarr deletion stopped halfway.

    Without ``zarr.json`` the dataset was converted (``zarr.json`` goes first,
    only after every check passed); what is left of the zarr goes once
    ``structures.atp`` and the table still match the recorded hash.

    Raises:
        ConversionError: if there are leftovers but ``structures.atp`` is
            missing or no longer matches the recorded hash.
    """
    leftovers = _present_zarr_entries(directory)
    if not leftovers:
        return  # already converted: unchanged
    structures_path = directory / STRUCTURES_FILENAME
    if not structures_path.exists():
        raise ConversionError(
            f"zarr entries {[path.name for path in leftovers]} without zarr.json "
            f"and without {STRUCTURES_FILENAME}"
        )
    if _current_structures_sha256(directory) != _recorded_structures_sha256(directory):
        raise ConversionError(
            f"zarr leftovers, and {STRUCTURES_FILENAME} does not match the recorded "
            "structures_sha256"
        )
    for path in leftovers:
        _remove(path)
    report.changes.append(
        f"deleted the zarr leftovers ({', '.join(path.name for path in leftovers)})"
    )


def make_shadow(directory: Path, shadow_root: Path) -> Path:
    """``<shadow_root>/<name>``: links to the source's entries, copies of the
    files the conversion rewrites. An existing shadow is reused (a rerun)."""
    shadow = shadow_root / directory.name
    if shadow.exists():
        return shadow
    shadow.mkdir(parents=True)
    for entry in directory.iterdir():
        if entry.name in REWRITTEN_FILES:
            shutil.copy2(entry, shadow / entry.name)
        elif entry.name == SELECTIONS_DIRECTORY and entry.is_dir():
            (shadow / entry.name).mkdir()
            for selection_file in entry.iterdir():
                if selection_file.suffix == ".yaml":
                    shutil.copy2(
                        selection_file, shadow / entry.name / selection_file.name
                    )
                else:
                    (shadow / entry.name / selection_file.name).symlink_to(
                        selection_file.resolve()
                    )
        else:
            (shadow / entry.name).symlink_to(entry.resolve())
    return shadow


def convert_dataset(
    directory: Path,
    *,
    dry_run: bool,
    keep_zarr: bool,
    workers: int,
    shadow_root: Path | None = None,
) -> DatasetConversionReport:
    """Convert one dataset (or its shadow on a dry run) and verify the result.

    Failures are reported, not raised.
    """
    started = time.monotonic()
    directory = Path(directory)
    report = DatasetConversionReport(dataset_path=directory, outcome="unchanged")
    try:
        if dry_run:
            if shadow_root is None:
                raise ConversionError("a dry run needs a shadow root")
            target = make_shadow(directory, Path(shadow_root))
        else:
            target = directory
        report.converted_path = target
        convert_layout(target, report, keep_zarr=keep_zarr)
        verify_started = time.monotonic()
        verified = verify_dataset(target, workers=workers)
        report.verify_seconds = time.monotonic() - verify_started
        report.problems += verified.problems
        report.structures_sha256 = verified.structures_sha256
        if not report.structures and (target / STRUCTURES_FILENAME).exists():
            structure_file = StructureFile.open(target / STRUCTURES_FILENAME)
            report.structures = len(structure_file)
            report.structures_file_bytes = (target / STRUCTURES_FILENAME).stat().st_size
    except Exception as error:  # one dataset must not stop the others
        logger.exception("%s could not be converted", directory)
        report.problems.append(f"{type(error).__name__}: {error}")
    if report.problems:
        report.outcome = "failed"
    elif report.changes:
        report.outcome = "would_convert" if dry_run else "converted"
    report.elapsed_seconds = time.monotonic() - started
    return report


def convert(
    paths: list[Path],
    *,
    dry_run: bool,
    keep_zarr: bool,
    workers: int,
    shadow_root: Path | None = None,
) -> ConversionReport:
    """Convert every dataset the paths stand for, one after the other.

    Raises:
        ConversionError: if a path is neither a dataset nor holds any.
    """
    report = ConversionReport(dry_run=dry_run, keep_zarr=keep_zarr)
    for directory in dataset_directories(paths):
        dataset_report = convert_dataset(
            directory,
            dry_run=dry_run,
            keep_zarr=keep_zarr,
            workers=workers,
            shadow_root=shadow_root,
        )
        logger.info(
            "%s: %s in %.1f s%s",
            directory,
            dataset_report.outcome,
            dataset_report.elapsed_seconds,
            "".join(f"\n  {line}" for line in dataset_report.changes)
            + "".join(f"\n  problem: {line}" for line in dataset_report.problems),
        )
        report.datasets.append(dataset_report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="dataset directories, or directories whose children are datasets",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="convert and verify a shadow of each dataset under --shadow-root",
    )
    parser.add_argument(
        "--shadow-root", type=Path, help="where a dry run builds its shadows"
    )
    parser.add_argument(
        "--keep-zarr",
        action="store_true",
        help="leave the zarr arrays in place after a checked conversion",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="processes for verify's geometry checks (default: the job's CPUs)",
    )
    parser.add_argument("--report", type=Path, help="also write the report as yaml")
    arguments = parser.parse_args(argv)
    if arguments.dry_run and arguments.shadow_root is None:
        parser.error("--dry-run needs --shadow-root")
    setup_logging()
    report = convert(
        arguments.paths,
        dry_run=arguments.dry_run,
        keep_zarr=arguments.keep_zarr,
        workers=arguments.workers,
        shadow_root=arguments.shadow_root,
    )
    if arguments.report is not None:
        write_model(arguments.report, report)
    outcomes: list[Outcome] = ["unchanged", "converted", "would_convert", "failed"]
    print(", ".join(f"{report.count(outcome)} {outcome}" for outcome in outcomes))
    for dataset in report.datasets:
        if dataset.outcome == "failed":
            print(f"FAILED {dataset.dataset_path}: {'; '.join(dataset.problems)}")
    return 1 if report.count("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
