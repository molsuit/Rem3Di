"""Tests for the ablation job-submission helpers.

Covers config snapshotting and one-job-per-config submission in
``scripts/submit_job.py``. No SLURM, no GPU: submission is exercised with
``--no-submit`` or a recorded ``subprocess.run``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pydantic_yaml as pyaml

from remedi.configuration.dataloader_config import (
    BucketBatchSamplingConfig,
    DataLoaderConfig,
)
from remedi.configuration.training_config import PretrainingConfig
from remedi.training.data.mixture import (
    DatasetComponent,
    MoleculeFractionValidation,
    PretrainingDataConfig,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    """Import a module from the (non-package) scripts/ directory by path."""
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


submit_job = _load_script("submit_job.py")


def _write_training_config(
    tmp_path: Path, name: str, *, output_base: Path, dataset_path: Path
) -> Path:
    """Write a minimal-but-valid PretrainingConfig yaml plus a dummy arch file."""
    arch = tmp_path / f"{name}_arch.yaml"
    arch.write_text("# dummy architecture config (only copied, not parsed here)\n")

    config = PretrainingConfig(
        training_name=name,
        dataloader=DataLoaderConfig(
            batch_sampling=BucketBatchSamplingConfig(max_atoms_per_batch=10500),
            num_workers=0,
        ),
        epochs=1,
        learning_rate=3e-4,
        weight_decay=1e-3,
        data=PretrainingDataConfig(
            domain="drug_like",
            components=[DatasetComponent(dataset=dataset_path)],
            validation=MoleculeFractionValidation(fraction=0.02),
        ),
        noise_level=0.3,
        model_config_path=arch,
        output_base=output_base,
    )
    cfg_path = tmp_path / f"{name}.yaml"
    pyaml.to_yaml_file(cfg_path, config)
    return cfg_path


def test_snapshot_freezes_config_and_arch(tmp_path: Path) -> None:
    output_base = tmp_path / "runs"
    cfg = _write_training_config(
        tmp_path, "run_a", output_base=output_base, dataset_path=tmp_path / "ds"
    )

    snapshot, name = submit_job.snapshot_configs(cfg)

    assert name == "run_a"
    assert snapshot.exists()
    snap_cfg = pyaml.parse_yaml_file_as(PretrainingConfig, snapshot)
    # Snapshot is frozen inside its own training directory...
    assert snap_cfg.training_directory == snapshot.parent
    # ...and the architecture config travels with it.
    assert snap_cfg.model_config_path == snapshot.parent / "architecture_config.yaml"
    assert snap_cfg.model_config_path.exists()


def test_no_submit_snapshots_every_config(tmp_path: Path, monkeypatch, capsys) -> None:
    output_base = tmp_path / "runs"
    configs = [
        _write_training_config(
            tmp_path, f"run_{i}", output_base=output_base, dataset_path=tmp_path / "ds"
        )
        for i in range(5)
    ]

    # sbatch must never be called in --no-submit mode.
    def _no_sbatch(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("sbatch should not run with --no-submit")

    monkeypatch.setattr(submit_job.subprocess, "run", _no_sbatch)
    monkeypatch.setattr(
        submit_job.sys, "argv", ["submit_job.py", *map(str, configs), "--no-submit"]
    )

    submit_job.main()

    # 5 runs snapshotted -> 5 training directories under output_base.
    assert len(list(output_base.glob("*/"))) == 5
    # Stdout lists every snapshot path (one per run).
    printed = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(printed) == 5
    assert all(Path(p).exists() for p in printed)


def test_one_job_is_submitted_per_config(tmp_path: Path, monkeypatch) -> None:
    output_base = tmp_path / "runs"
    configs = [
        _write_training_config(
            tmp_path, f"run_{i}", output_base=output_base, dataset_path=tmp_path / "ds"
        )
        for i in range(3)
    ]
    commands = []
    monkeypatch.setattr(
        submit_job.subprocess, "run", lambda command, **_: commands.append(command)
    )
    monkeypatch.setattr(
        submit_job.sys,
        "argv",
        ["submit_job.py", *map(str, configs), "--group", "g", "--time", "02:00:00"],
    )
    submit_job.main()
    assert [command[1] for command in commands] == [
        f"--job-name=g_run_{i}" for i in range(3)
    ]
    assert all(command[2] == "--time=02:00:00" for command in commands)
    assert all(command[3] == "scripts/pretrain.slurm" for command in commands)
