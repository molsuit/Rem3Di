"""Dump the curated Polaris hub datasets to plain parquets.

Step one of two. ``polaris-lib`` pins ``zarr<3`` whereas this project requires
``zarr>=3.2``, so the two cannot share a Python env. This script declares its
dependencies inline (PEP 723), ``uv run`` materialises an ephemeral polaris-only
env for it, and the project env is never touched. Step two,
``prepare-polaris`` (``remedi_prepare_polaris/prepare.py``), runs in the project
env and turns the parquets into bundles without ever calling polaris.

Usage, from the repository root::

    uv run preparers/polaris/dump_polaris.py

    # Re-download every dataset even if its parquet already exists.
    uv run preparers/polaris/dump_polaris.py --force

For each entry in ``DATASETS`` two files are written under ``--out-root``
(default: ``benchmark_data/raw/polaris`` of this repository):

``<dataset_id>.parquet``
    * ``smiles`` -- str, the source SMILES verbatim (a CXSMILES keeps its
      ``|...|`` extension, enhanced stereo groups included)
    * ``split`` -- uint8 split code (0=train, 1=valid, 2=test, 255=unassigned),
      from the dataset's ``Set`` column; 255 everywhere when it has none
    * one float column per source task, named verbatim

``<dataset_id>.source.yaml``
    The slug, the ``polaris-lib`` version, the hub's checksum of the dataset
    where it publishes one, and the dump time. The preparer hashes both files
    into each bundle's provenance.

Existing parquets are skipped (idempotent re-runs); pass ``--force`` to re-dump.
Polaris benchmark objects are intentionally not used: they are scored subsets
of the underlying datasets, so the dataset is fetched directly and the preparer's
``datasets.yaml`` picks the label columns.
"""

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polaris-lib>=0.13",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "pyyaml",
# ]
# ///

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as installed_version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("dump_polaris")

#: ``Rem3Di/preparers/polaris/dump_polaris.py`` -> ``Rem3Di``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPOSITORY_ROOT / "benchmark_data" / "raw" / "polaris"

# Split codes match remedi.data_handling.dataset.tasks.Split. Hardcoded (not
# imported) so this script stays free of the remedi package.
SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST, SPLIT_UNASSIGNED = 0, 1, 2, 255

#: Polaris ``Set`` labels seen across the ASAP / Biogen / Polaris datasets.
SET_LABEL_TO_CODE = {
    "train": SPLIT_TRAIN,
    "valid": SPLIT_VALID,
    "validation": SPLIT_VALID,
    "val": SPLIT_VALID,
    "test": SPLIT_TEST,
}

#: Columns that are never task targets (metadata, identifiers, the split).
NON_TASK_COLUMNS = {
    "Molecule Name",
    "Set",
    "CXSMILES",
    "smiles",
    "SMILES",
    "Smiles",
    "MOL_smiles",
    "UNIQUE_ID",
    "MOL_smiles_index",
    # Polaris bookkeeping that occasionally surfaces as a column:
    "MOL_molhash_id",
}


@dataclass(frozen=True)
class PolarisDataset:
    """One hub dataset to dump."""

    dataset_id: str
    slug: str
    smiles_column: str


#: The curated Polaris datasets. The preparer's ``datasets.yaml`` maps each
#: parquet to one or more bundles.
DATASETS: tuple[PolarisDataset, ...] = (
    PolarisDataset(
        "polaris_antiviral_admet",
        "asap-discovery/antiviral-admet-2025-unblinded",
        "CXSMILES",
    ),
    PolarisDataset(
        "polaris_antiviral_potency",
        "asap-discovery/antiviral-potency-2025-unblinded",
        "CXSMILES",
    ),
    PolarisDataset("polaris_adme_fang", "biogen/adme-fang-v1", "MOL_smiles"),
    PolarisDataset(
        "polaris_pkis2_subset", "polaris/drewry2017-pkis2-subset-v2", "MOL_smiles"
    ),
)


def split_codes_from_set(set_column: pd.Series) -> np.ndarray:
    codes = np.full(len(set_column), SPLIT_UNASSIGNED, dtype=np.uint8)
    if set_column.empty:
        return codes
    normalized = set_column.astype(str).str.strip().str.lower()
    for label, code in SET_LABEL_TO_CODE.items():
        codes[(normalized == label).to_numpy()] = code
    return codes


def load_polaris_table(slug: str) -> tuple[pd.DataFrame, list[str], str | None]:
    """Fetch a Polaris dataset as a DataFrame, its column list and the hub's checksum.

    The checksum is the table md5 of a ``DatasetV1`` or the zarr manifest md5 of
    a ``DatasetV2``; ``None`` when the hub publishes neither. Polaris is imported
    lazily so the rest of the file can be byte-compiled in the main project venv
    (which intentionally lacks ``polaris-lib``).
    """
    import polaris
    from polaris.dataset import DatasetV1, DatasetV2

    dataset = polaris.load_dataset(slug)
    columns = list(dataset.columns)
    if isinstance(dataset, DatasetV1):
        return pd.DataFrame(dataset.table[:]), columns, dataset.md5sum
    if isinstance(dataset, DatasetV2):
        frame = pd.DataFrame(
            {column: dataset.zarr_data[column][:] for column in columns}
        )
        checksum = (
            dataset.zarr_manifest_md5sum if dataset.has_zarr_manifest_md5sum else None
        )
        return frame, columns, checksum
    raise SystemExit(f"Unexpected polaris dataset class: {type(dataset).__name__}")


def dump(dataset: PolarisDataset, out_root: Path) -> None:
    """Write ``<dataset_id>.parquet`` and ``<dataset_id>.source.yaml``."""
    frame, columns, checksum = load_polaris_table(dataset.slug)

    if dataset.smiles_column not in columns:
        raise SystemExit(
            f"{dataset.dataset_id}: SMILES column {dataset.smiles_column!r} not "
            f"found. Available: {columns}"
        )

    if "Set" in columns:
        split_codes = split_codes_from_set(frame["Set"])
    else:
        logger.warning(
            "%s: no 'Set' column; every row is written as unassigned (255) and "
            "the preparer derives the split.",
            dataset.dataset_id,
        )
        split_codes = np.full(len(frame), SPLIT_UNASSIGNED, dtype=np.uint8)

    task_columns = [column for column in columns if column not in NON_TASK_COLUMNS]
    if not task_columns:
        raise SystemExit(
            f"{dataset.dataset_id}: no task columns inferred from {columns!r}. "
            "Update NON_TASK_COLUMNS in this script."
        )

    out_frame = pd.DataFrame(
        {"smiles": frame[dataset.smiles_column].astype(str).to_numpy()}
    )
    out_frame["split"] = split_codes
    for column in task_columns:
        out_frame[column] = pd.to_numeric(frame[column], errors="coerce")

    out_root.mkdir(parents=True, exist_ok=True)
    parquet_path = out_root / f"{dataset.dataset_id}.parquet"
    out_frame.to_parquet(parquet_path, index=False)

    source_record = {
        "slug": dataset.slug,
        "polaris_lib_version": installed_version("polaris-lib"),
        "checksum": checksum,
        "dumped_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    source_path = out_root / f"{dataset.dataset_id}.source.yaml"
    source_path.write_text(yaml.safe_dump(source_record, sort_keys=False))

    counts = {
        "train": int((split_codes == SPLIT_TRAIN).sum()),
        "valid": int((split_codes == SPLIT_VALID).sum()),
        "test": int((split_codes == SPLIT_TEST).sum()),
        "unassigned": int((split_codes == SPLIT_UNASSIGNED).sum()),
    }
    logger.info(
        "%s: %d rows -> %s (splits=%s, %d tasks)",
        dataset.dataset_id,
        len(out_frame),
        parquet_path,
        counts,
        len(task_columns),
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Directory receiving <dataset_id>.parquet and <dataset_id>.source.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-dump even if the parquet already exists (default: skip).",
    )
    arguments = parser.parse_args()

    for dataset in DATASETS:
        parquet_path = arguments.out_root / f"{dataset.dataset_id}.parquet"
        if parquet_path.exists() and not arguments.force:
            logger.info(
                "%s: %s exists; skipping (pass --force to re-dump).",
                dataset.dataset_id,
                parquet_path,
            )
            continue
        dump(dataset, arguments.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
