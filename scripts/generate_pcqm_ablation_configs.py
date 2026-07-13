"""Regenerate the PCQM pretraining ablation with VICReg disabled.

The previous ablation (``/p/scratch/mace/wedig1/training_runs/pcqm_ablation``)
swept 11 architecture/training knobs against a single baseline *with* the VICReg
variance + covariance regularizer enabled. This rebuilds the exact same 11 runs
but:

* VICReg is **completely disabled** (the crucial difference -- the descriptor is
  now shaped purely by the atom-denoising objective).
* dataset -> the 500k JUWELS subset (``pcqm500k_std``),
* MACE model / output paths -> JUWELS,
* W&B group -> ``pcqm_ablation_novicreg``.

Every other knob is copied verbatim from each source run's saved
``architecture_config.yaml`` / ``training_config.yaml``, so the only intended
differences between the two ablations are the four bullets above.

Configs are written into the repo at
``configs/training/pcqm_ablation_novicreg/<run>/`` (one dir per run, each with a
``training_config.yaml`` + ``architecture_config.yaml``). Submit them packed
four-per-node with::

    uv run python scripts/submit_job.py \\
        configs/training/pcqm_ablation_novicreg/*/training_config.yaml \\
        --group pcqm_novicreg
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pydantic_yaml as pyaml

from remedi.configuration.architecture_config import ArchitectureConfig
from remedi.configuration.dataloader_config import BucketBatchSamplingConfig
from remedi.configuration.training_config import TrainingConfig

SOURCE_ABLATION = Path("/p/scratch/mace/wedig1/training_runs/pcqm_ablation")
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "configs" / "training" / "pcqm_ablation_novicreg"

MACE_MODEL = REPO_ROOT / "MACE-POLAR-1-M.model"
DATASET_PATH = Path("/p/scratch/mace/wedig1/datasets/pcqm4m/pcqm500k_std")
OUTPUT_BASE = Path("/p/scratch/mace/wedig1/training_runs/pcqm_ablation_novicreg")
RUN_GROUP = "pcqm_ablation_novicreg"
EPOCHS = 15

# Strip the "NN-DATE-" snapshot prefix from a source run dir name.
_PREFIX = re.compile(r"^\d+-[\d_]+-")


def clean_run_name(source_dir_name: str) -> str:
    return _PREFIX.sub("", source_dir_name)


def source_run_dirs() -> list[Path]:
    dirs = [d for d in SOURCE_ABLATION.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No run directories under {SOURCE_ABLATION}")
    return sorted(dirs, key=lambda d: d.name)


def build_run(source_dir: Path) -> str:
    """Materialise one vicreg-disabled run dir; returns the clean run name."""
    name = clean_run_name(source_dir.name)
    run_dir = OUT_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Architecture: copy verbatim, only repoint the MACE model to JUWELS. Read
    # the *post-training* config: it has the MACE irrep signature already
    # resolved/serialized, so parsing it here does not load MACE (no GPU on the
    # login node). The pre-training config leaves those None, which would force
    # a CUDA model load on parse.
    arch = pyaml.parse_yaml_file_as(
        ArchitectureConfig, source_dir / "post_training_architecture_config.yaml"
    )
    assert arch.mace_config is not None
    arch.mace_config.model_path = MACE_MODEL
    arch_path = run_dir / "architecture_config.yaml"
    pyaml.to_yaml_file(arch_path, arch)

    # Training: copy verbatim, disable VICReg, repoint dataset + outputs.
    train = pyaml.parse_yaml_file_as(
        TrainingConfig, source_dir / "training_config.yaml"
    )
    train.vicreg.enabled = False
    train.epochs = EPOCHS
    # Halve the per-batch atom budget: the prior ablation ran on 96 GB cards,
    # the JUWELS Booster A100s have only 40 GB.
    sampling = train.dataloader.batch_sampling
    if isinstance(sampling, BucketBatchSamplingConfig):
        sampling.max_atoms_per_batch //= 2
    train.dataset_path = DATASET_PATH
    train.output_base = OUTPUT_BASE
    train.run_group = RUN_GROUP
    train.model_config_path = arch_path
    # Cleared so submit_job.py snapshots into a fresh frozen directory.
    train.training_directory = None
    train_path = run_dir / "training_config.yaml"
    pyaml.to_yaml_file(train_path, train)

    return name


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    names = [build_run(d) for d in source_run_dirs()]
    print(f"Wrote {len(names)} run(s) to {OUT_DIR}:")
    for name in names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
