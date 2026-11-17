"""The one-off conversion of zarr datasets to ``structures.atp``.

``scripts/dataset_creation/convert_zarr_to_atompack.py`` reads the zarr arrays,
writes ``structures.atp``, compares it exactly, re-records ``structures_sha256``
and deletes the zarr. The zarr layout is simulated here with ``zarr`` (the dev
dependency) on a dataset written by the current writer: the frames, charges and
ids go into sharded zarr arrays, ``structures.atp`` goes, and
``provenance.yaml`` records the zarr-era hash.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
import zarr
from zarr.codecs import BloscCodec

from remedi_data.dataset import Dataset
from remedi_data.format import (
    STRUCTURES_FILENAME,
    TABLE_FILENAME,
    EvalMetric,
    LabelColumn,
    TaskType,
    read_bundle,
    read_provenance,
)
from remedi_data.format.verify import verify_dataset
from remedi_data.selection import (
    RandomMolecules,
    SelectionConfig,
    read_selection,
    select_structures,
)
from testing import write_small_dataset

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(relative_path: str):
    path = SCRIPTS_DIR / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module  # pydantic looks the module up there
    spec.loader.exec_module(module)
    return module


conversion = _load_script("dataset_creation/convert_zarr_to_atompack.py")

ZARR_ENTRIES = {
    "positions",
    "atomic_numbers",
    "molecule_ptr",
    "total_charge",
    "multiplicity",
    "ids",
    "zarr.json",
    "dataset_config.yaml",
}


def _benchmark(root: Path, dataset_id: str = "benchmark") -> Path:
    return write_small_dataset(
        root,
        dataset_id=dataset_id,
        labels=[LabelColumn(name="potency", task_type=TaskType.regression)],
        metrics=[EvalMetric.mae],
        targets=np.linspace(0.0, 9.0, 10).reshape(-1, 1),
    )


def _zarr_era_sha256(directory: Path) -> str:
    """The definition of 2026-11-16: five arrays, each with a header."""
    digest = hashlib.sha256()
    for name in (
        "atomic_numbers",
        "positions",
        "molecule_ptr",
        "total_charge",
        "multiplicity",
    ):
        values = np.ascontiguousarray(np.asarray(zarr.open_array(directory / name)[:]))
        digest.update(f"{name}:{values.dtype.str}:{values.shape}\n".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _write_zarr_layout(directory: Path) -> None:
    """Turn a current dataset into what the zarr writer wrote (small shards, so
    every array spans several shard files)."""
    structures = Dataset.open(directory).read_structures()
    table = read_bundle(directory).table
    assert structures.total_charge is not None
    assert structures.multiplicity is not None
    group = zarr.create_group(store=directory, overwrite=False)
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")

    def write_array(parent, name: str, data: np.ndarray, chunk: int) -> None:
        parent.create_array(
            name=name,
            data=data,
            chunks=(chunk, *data.shape[1:]),
            shards=(chunk * 4, *data.shape[1:]),
            compressors=compressor,
        )

    write_array(group, "positions", structures.positions, 8)
    write_array(group, "atomic_numbers", structures.atomic_numbers, 8)
    write_array(group, "molecule_ptr", structures.ptr, 2)
    write_array(group, "total_charge", structures.total_charge, 2)
    write_array(group, "multiplicity", structures.multiplicity, 2)
    ids = group.require_group("ids")
    for name in ("structure_id", "molecule_id", "stereoisomer_id"):
        write_array(ids, name, table[name].to_numpy(dtype=np.int64), 2)
    (directory / "dataset_config.yaml").write_text(
        "atom_chunk: 8\nmolecule_chunk: 2\n"
        "atom_chunks_per_shard: 4\nmolecule_chunks_per_shard: 4\n"
    )
    (directory / STRUCTURES_FILENAME).unlink()
    zarr_era_hash = _zarr_era_sha256(directory)
    provenance_path = directory / "provenance.yaml"
    document = yaml.safe_load(provenance_path.read_text())
    document["outputs"]["structures_sha256"] = zarr_era_hash
    provenance_path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    )
    # A selection made in the zarr era records the zarr-era hash.
    for record_path in (directory / "selections").glob("*.yaml"):
        record = yaml.safe_load(record_path.read_text())
        record["structures_sha256"] = zarr_era_hash
        record_path.write_text(yaml.safe_dump(record, sort_keys=False))


@pytest.fixture
def zarr_dataset(tmp_path: Path) -> Path:
    directory = _benchmark(tmp_path / "datasets")
    select_structures(
        directory,
        SelectionConfig(name="half", steps=[RandomMolecules(count=5, seed=0)]),
    )
    np.save(tmp_path / "half.npy", read_selection(directory, "half"))
    expected = Dataset.open(directory).read_structures()
    np.savez(
        tmp_path / "expected.npz",
        **{
            "positions": expected.positions,
            "atomic_numbers": expected.atomic_numbers,
            "ptr": expected.ptr,
        },
    )
    _write_zarr_layout(directory)
    return directory


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_a_zarr_dataset_is_converted_checked_and_verified(
    zarr_dataset: Path, tmp_path: Path
) -> None:
    table_bytes = (zarr_dataset / TABLE_FILENAME).read_bytes()
    recorded = read_provenance(zarr_dataset).outputs
    assert recorded is not None

    report = conversion.convert(
        [zarr_dataset.parent], dry_run=False, keep_zarr=False, workers=1
    )

    (dataset_report,) = report.datasets
    assert dataset_report.outcome == "converted", dataset_report.problems
    assert dataset_report.structures == 10
    assert dataset_report.zarr_files > 8 and dataset_report.structures_file_bytes > 0
    assert {path.name for path in zarr_dataset.iterdir()} == {
        "dataset.yaml",
        "provenance.yaml",
        TABLE_FILENAME,
        STRUCTURES_FILENAME,
        "selections",
    }
    # The table is not rewritten; only the structures hash moves.
    assert (zarr_dataset / TABLE_FILENAME).read_bytes() == table_bytes
    provenance = read_provenance(zarr_dataset)
    assert provenance.outputs is not None
    assert provenance.outputs.table_parquet == recorded.table_parquet
    dataset = Dataset.open(zarr_dataset)
    assert provenance.outputs.structures_sha256 == dataset.structures_sha256()
    assert provenance.outputs.structures_sha256 != recorded.structures_sha256
    assert dataset_report.recorded_structures_sha256 == recorded.structures_sha256
    expected = np.load(tmp_path / "expected.npz")
    structures = dataset.read_structures()
    for name in ("positions", "atomic_numbers", "ptr"):
        np.testing.assert_array_equal(getattr(structures, name), expected[name])
    assert verify_dataset(zarr_dataset).problems == []
    # The selection made in the zarr era reads again: its hash was re-recorded.
    assert "selections/half.yaml: structures_sha256 re-recorded" in (
        dataset_report.changes
    )
    np.testing.assert_array_equal(
        read_selection(zarr_dataset, "half"), np.load(tmp_path / "half.npy")
    )

    again = conversion.convert(
        [zarr_dataset], dry_run=False, keep_zarr=False, workers=1
    )
    assert [entry.outcome for entry in again.datasets] == ["unchanged"]


def test_a_dry_run_converts_a_shadow_and_leaves_the_source_alone(
    zarr_dataset: Path, tmp_path: Path
) -> None:
    before = _snapshot(zarr_dataset)
    shadow_root = tmp_path / "shadow"

    report = conversion.convert(
        [zarr_dataset],
        dry_run=True,
        keep_zarr=False,
        workers=1,
        shadow_root=shadow_root,
    )

    (dataset_report,) = report.datasets
    assert dataset_report.outcome == "would_convert", dataset_report.problems
    assert _snapshot(zarr_dataset) == before
    shadow = shadow_root / zarr_dataset.name
    assert dataset_report.converted_path == shadow
    assert not ({path.name for path in shadow.iterdir()} & ZARR_ENTRIES)
    assert verify_dataset(shadow).problems == []
    assert len(read_selection(shadow, "half")) > 0
    assert (shadow / "selections" / "half.parquet").is_symlink()
    again = conversion.convert(
        [zarr_dataset],
        dry_run=True,
        keep_zarr=False,
        workers=1,
        shadow_root=shadow_root,
    )
    assert [entry.outcome for entry in again.datasets] == ["unchanged"]
    assert _snapshot(zarr_dataset) == before


def test_keep_zarr_then_a_rerun_deletes_it(zarr_dataset: Path) -> None:
    kept = conversion.convert([zarr_dataset], dry_run=False, keep_zarr=True, workers=1)
    assert [entry.outcome for entry in kept.datasets] == ["converted"]
    assert ZARR_ENTRIES <= {path.name for path in zarr_dataset.iterdir()}
    assert (zarr_dataset / STRUCTURES_FILENAME).is_file()
    assert verify_dataset(zarr_dataset).problems == []

    # The existing structures.atp is compared with the zarr, not rewritten.
    written = (zarr_dataset / STRUCTURES_FILENAME).stat().st_mtime_ns
    finished = conversion.convert(
        [zarr_dataset], dry_run=False, keep_zarr=False, workers=1
    )
    (dataset_report,) = finished.datasets
    assert dataset_report.outcome == "converted", dataset_report.problems
    assert (zarr_dataset / STRUCTURES_FILENAME).stat().st_mtime_ns == written
    assert not ({path.name for path in zarr_dataset.iterdir()} & ZARR_ENTRIES)


def test_a_deletion_that_stopped_halfway_is_finished(zarr_dataset: Path) -> None:
    conversion.convert([zarr_dataset], dry_run=False, keep_zarr=True, workers=1)
    (zarr_dataset / "zarr.json").unlink()  # deletion goes zarr.json first

    report = conversion.convert(
        [zarr_dataset], dry_run=False, keep_zarr=False, workers=1
    )

    (dataset_report,) = report.datasets
    assert dataset_report.outcome == "converted", dataset_report.problems
    assert "leftovers" in dataset_report.changes[0]
    assert not ({path.name for path in zarr_dataset.iterdir()} & ZARR_ENTRIES)


def _change_a_molecule_id_in_the_zarr(directory: Path) -> None:
    array = zarr.open_array(directory / "ids" / "molecule_id", mode="r+")
    values = np.asarray(array[:])
    values[0] = values.max() + 1
    array[:] = values


def _move_an_atom_in_the_zarr(directory: Path) -> None:
    array = zarr.open_array(directory / "positions", mode="r+")
    values = np.asarray(array[:])
    values[0, 0] += 0.5
    array[:] = values


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        (_change_a_molecule_id_in_the_zarr, "molecule_id differs"),
        (_move_an_atom_in_the_zarr, "does not match the structures_sha256"),
    ],
)
def test_a_dataset_that_fails_a_check_is_left_as_it_was(
    zarr_dataset: Path, damage, expected: str
) -> None:
    damage(zarr_dataset)
    before = _snapshot(zarr_dataset)

    report = conversion.convert(
        [zarr_dataset], dry_run=False, keep_zarr=False, workers=1
    )

    (dataset_report,) = report.datasets
    assert dataset_report.outcome == "failed"
    assert any(expected in problem for problem in dataset_report.problems)
    assert _snapshot(zarr_dataset) == before


def test_the_command_line_writes_a_report(zarr_dataset: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "report.yaml"
    assert (
        conversion.main(
            [str(zarr_dataset), "--workers", "1", "--report", str(report_path)]
        )
        == 0
    )
    document = yaml.safe_load(report_path.read_text())
    assert [entry["outcome"] for entry in document["datasets"]] == ["converted"]
    with pytest.raises(SystemExit):
        conversion.main([str(zarr_dataset), "--dry-run"])
