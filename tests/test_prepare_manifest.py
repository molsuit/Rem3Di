"""The prepare manifest (``BENCHMARK_DATA_FORMAT.md`` §3).

Yaml round trip, the discriminated union on ``kind``, per-dataset task
expansion, and the fault isolation that expansion buys: one dataset failing is
one failed entry in ``status.yaml`` while the rest of the panel finishes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydantic_yaml as pyd_yaml
import pytest
import yaml
from pydantic import ValidationError

from remedi.data_handling.bundle import BenchmarkTask, read_bundle
from remedi.data_handling.dataset.tasks import TaskType
from remedi.data_handling.prepare import (
    GenerateConformersConfig,
    IngestBenchmarkConfig,
    PrepareContext,
    PrepareManifest,
    VerifyBenchmarkConfig,
    prepare,
)
from remedi.evaluation.framework.task_runner import RunReport

from .helpers.bundle_fixtures import (
    write_conformers_bundle,
    write_smiles_bundle,
)


def build_manifest(
    tmp_path: Path, dataset_ids: list[str] | None = None
) -> PrepareManifest:
    return PrepareManifest(
        smiles_bundle_root=tmp_path / "bundles",
        benchmark_root=tmp_path / "benchmark_bundles",
        output_root=tmp_path / "prepare_out",
        tasks=[GenerateConformersConfig(dataset_ids=dataset_ids, n_workers=1)],
    )


# ------------------------------------------------------------------ the model


def test_manifest_round_trips_through_yaml(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path, dataset_ids=["esol", "bbb"])
    path = tmp_path / "manifest.yaml"
    pyd_yaml.to_yaml_file(path, manifest)

    reloaded = pyd_yaml.parse_yaml_file_as(PrepareManifest, path)

    assert reloaded == manifest
    (task,) = reloaded.tasks
    assert isinstance(task, GenerateConformersConfig)
    assert task.kind == "generate_conformers"
    assert task.geometry_limits.max_atoms is None
    assert task.geometry_limits.min_interatomic_distance == 0.5
    assert task.geometry_limits.allowed_element_symbols() is not None


def test_manifest_is_readable_from_a_hand_written_yaml(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "smiles_bundle_root": str(tmp_path / "bundles"),
                "benchmark_root": str(tmp_path / "benchmark_bundles"),
                "output_root": str(tmp_path / "prepare_out"),
                "tasks": [{"kind": "generate_conformers", "n_conformers": 2}],
            }
        )
    )

    manifest = pyd_yaml.parse_yaml_file_as(PrepareManifest, path)

    (task,) = manifest.tasks
    assert isinstance(task, GenerateConformersConfig)
    assert task.n_conformers == 2
    assert task.dataset_ids is None
    assert manifest.keep_going is True
    assert manifest.seed == 0


def test_the_shipped_template_parses() -> None:
    template = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "dataset_creation"
        / "prepare_benchmarks_template.yaml"
    )
    document = yaml.safe_load(template.read_text())

    manifest = PrepareManifest.model_validate(document)

    generate, ingest, verify = manifest.tasks
    assert isinstance(generate, GenerateConformersConfig)
    assert generate.max_embed_attempts == 200
    assert generate.max_mmff_steps == 100
    assert generate.mmff_non_bonded_threshold == 100.0
    assert isinstance(ingest, IngestBenchmarkConfig)
    assert ingest.overwrite is False
    assert isinstance(verify, VerifyBenchmarkConfig)
    assert verify.check_zarr is True


def test_each_task_kind_resolves_against_the_root_it_reads(tmp_path: Path) -> None:
    """The three kinds discover from different roots, and expansion knows which."""
    write_smiles_bundle(tmp_path / "bundles", dataset_id="from_smiles")
    write_conformers_bundle(
        tmp_path / "benchmark_bundles",
        dataset_id="from_conformers",
        tasks=[BenchmarkTask(name="y", task_type=TaskType.regression)],
        metrics=["RMSE"],
        targets=np.linspace(0.0, 1.0, 10),
    )
    manifest = PrepareManifest(
        smiles_bundle_root=tmp_path / "bundles",
        benchmark_root=tmp_path / "benchmark_bundles",
        output_root=tmp_path / "prepare_out",
        tasks=[
            GenerateConformersConfig(n_workers=1),
            IngestBenchmarkConfig(),
            VerifyBenchmarkConfig(),
        ],
    )

    expanded = manifest.expand_tasks()

    assert [(task.kind, task.dataset_ids) for task in expanded] == [
        ("generate_conformers", ["from_smiles"]),
        ("ingest_benchmark", ["from_conformers"]),
        ("verify_benchmark", ["from_conformers"]),
    ]


def test_ingest_and_verify_run_through_the_manifest(tmp_path: Path) -> None:
    write_conformers_bundle(
        tmp_path / "benchmark_bundles",
        dataset_id="toy",
        tasks=[BenchmarkTask(name="y", task_type=TaskType.regression)],
        metrics=["RMSE"],
        targets=np.linspace(0.0, 1.0, 10),
    )
    manifest = PrepareManifest(
        smiles_bundle_root=tmp_path / "bundles",
        benchmark_root=tmp_path / "benchmark_bundles",
        output_root=tmp_path / "prepare_out",
        tasks=[IngestBenchmarkConfig(), VerifyBenchmarkConfig()],
    )

    report = prepare(manifest)

    assert (report.n_tasks, report.n_failed) == (2, 0)
    assert [entry.name for entry in report.statuses] == [
        "0_ingest_benchmark_toy",
        "1_verify_benchmark_toy",
    ]
    # The zarr and both per-dataset summaries landed under output_root, and the
    # bundle travelled with the zarr so the eval side can discover it.
    zarr_path = manifest.output_root / "toy"
    assert (zarr_path / "dataset_config.yaml").is_file()
    assert (zarr_path / "benchmark.yaml").is_file()
    assert (zarr_path / "table.parquet").is_file()
    assert (manifest.output_root / "ingest_benchmark" / "toy.yaml").is_file()
    assert (manifest.output_root / "verify_benchmark" / "toy.yaml").is_file()


def test_manifest_rejects_unknown_fields_and_empty_task_lists(tmp_path: Path) -> None:
    base = build_manifest(tmp_path).model_dump(mode="json")
    with pytest.raises(ValidationError):
        PrepareManifest.model_validate({**base, "benchmark_toot": "typo"})
    with pytest.raises(ValidationError):
        PrepareManifest.model_validate({**base, "tasks": []})
    with pytest.raises(ValidationError):
        PrepareManifest.model_validate({**base, "tasks": [{"kind": "not_a_task"}]})


def test_prepare_context_task_dir_is_created(tmp_path: Path) -> None:
    ctx = PrepareContext(
        smiles_bundle_root=tmp_path / "bundles",
        benchmark_root=tmp_path / "benchmark_bundles",
        output_root=tmp_path / "prepare_out",
    )
    directory = ctx.task_dir("generate_conformers")
    assert directory == tmp_path / "prepare_out" / "generate_conformers"
    assert directory.is_dir()


# --------------------------------------------------------------- the expansion


def test_expand_tasks_gives_one_task_per_discovered_dataset(tmp_path: Path) -> None:
    for dataset_id in ("zeta", "alpha"):
        write_smiles_bundle(tmp_path / "bundles", dataset_id=dataset_id)
    manifest = build_manifest(tmp_path)

    expanded = manifest.expand_tasks()

    assert [task.dataset_ids for task in expanded] == [["alpha"], ["zeta"]]
    assert [task.status_label for task in expanded] == ["alpha", "zeta"]
    # Every other knob is copied verbatim onto each per-dataset task.
    assert all(task.n_workers == 1 for task in expanded)
    assert manifest.resolved_dataset_ids() == ["alpha", "zeta"]


def test_expand_tasks_keeps_an_explicit_dataset_list(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path, dataset_ids=["esol", "bbb"])
    assert [task.dataset_ids for task in manifest.expand_tasks()] == [
        ["esol"],
        ["bbb"],
    ]


def test_expand_tasks_over_an_empty_root_is_empty(tmp_path: Path) -> None:
    assert build_manifest(tmp_path).expand_tasks() == []


# ------------------------------------------------------------ fault isolation


def test_status_yaml_has_one_entry_per_dataset(tmp_path: Path) -> None:
    for dataset_id in ("alpha", "zeta"):
        write_smiles_bundle(tmp_path / "bundles", dataset_id=dataset_id)
    manifest = build_manifest(tmp_path)

    report = prepare(manifest)

    assert report.n_tasks == 2
    assert report.n_failed == 0
    status = pyd_yaml.parse_yaml_file_as(
        RunReport, manifest.output_root / "status.yaml"
    )
    assert [entry.name for entry in status.statuses] == [
        "0_generate_conformers_alpha",
        "1_generate_conformers_zeta",
    ]
    assert all(entry.ok for entry in status.statuses)
    assert all(entry.kind == "generate_conformers" for entry in status.statuses)
    assert (manifest.output_root / "manifest.yaml").is_file()
    for entry in status.statuses:
        (artifact,) = entry.artifacts
        assert (manifest.output_root / artifact["file_name"]).is_file()


def test_a_failing_dataset_is_isolated_from_a_good_one(tmp_path: Path) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id="good")
    # A directory that discovery accepts (it has a benchmark.yaml) but that
    # read_bundle refuses: the table is missing.
    broken = tmp_path / "bundles" / "broken"
    broken.mkdir(parents=True)
    (broken / "benchmark.yaml").write_text(
        (tmp_path / "bundles" / "good" / "benchmark.yaml").read_text()
    )
    manifest = build_manifest(tmp_path)

    report = prepare(manifest)

    assert report.n_tasks == 2
    assert report.n_failed == 1
    by_name = {entry.name: entry for entry in report.statuses}
    assert by_name["0_generate_conformers_broken"].ok is False
    assert by_name["0_generate_conformers_broken"].traceback is not None
    assert by_name["1_generate_conformers_good"].ok is True

    # The healthy dataset still produced a readable conformers-stage bundle.
    bundle = read_bundle(manifest.benchmark_root / "good")
    assert bundle.spec.stage == "conformers"
    assert not (manifest.benchmark_root / "broken").exists()

    # status.yaml on disk says the same thing.
    status = pyd_yaml.parse_yaml_file_as(
        RunReport, manifest.output_root / "status.yaml"
    )
    assert status.n_failed == 1


def test_keep_going_false_stops_at_the_first_failure_but_keeps_the_status(
    tmp_path: Path,
) -> None:
    write_smiles_bundle(tmp_path / "bundles", dataset_id="good")
    broken = tmp_path / "bundles" / "broken"
    broken.mkdir(parents=True)
    (broken / "benchmark.yaml").write_text(
        (tmp_path / "bundles" / "good" / "benchmark.yaml").read_text()
    )
    manifest = build_manifest(tmp_path)
    manifest.keep_going = False

    with pytest.raises(FileNotFoundError):
        prepare(manifest)

    status = pyd_yaml.parse_yaml_file_as(
        RunReport, manifest.output_root / "status.yaml"
    )
    assert [entry.name for entry in status.statuses] == ["0_generate_conformers_broken"]
    assert status.n_failed == 1
    assert status.n_tasks == 2
