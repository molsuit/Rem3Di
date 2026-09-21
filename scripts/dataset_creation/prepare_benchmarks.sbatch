#!/usr/bin/env bash
#SBATCH --job-name=prepare_benchmarks
#SBATCH --account=your_account
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/datasets/logs/slurm-%j.out
#SBATCH --error=/path/to/datasets/logs/slurm-%j.err
#
# Runs one prepare manifest over the benchmark panel: generate_conformers
# (bundle -> bundle), ingest_benchmark (bundle -> zarr) and verify_benchmark.
# Every task is expanded to one per dataset, so a wall-clock timeout mid-panel
# keeps the datasets that finished and `status.yaml` says which those were.
# Re-running is idempotent: a dataset whose conformers bundle (or zarr) already
# exists is skipped unless its task sets `overwrite: true`.
#
# Submit with:
#     sbatch scripts/dataset_creation/prepare_benchmarks.sbatch

set -euo pipefail

cd /path/to/Rem3Di
mkdir -p /path/to/datasets/logs

exec uv run python scripts/dataset_creation/prepare_benchmarks.py \
    --config configs/dataset_creation/prepare_benchmarks_local.yaml
