#!/usr/bin/env bash
#SBATCH --job-name=benchmark_build
#SBATCH --account=your_account
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/datasets/logs/slurm-%j.out
#SBATCH --error=/path/to/datasets/logs/slurm-%j.err
#
# Builds the full benchmark panel (every registered MoleculeNet + TDC ADMET
# dataset). Re-running is idempotent: zarrs whose directory already exists
# and contains a manifest are skipped; the user clears partial dirs by hand.
#
# Submit with:
#     sbatch scripts/dataset_creation/build_benchmark_dataset.sbatch

set -euo pipefail

cd /path/to/3DMolecularDescriptors
mkdir -p /path/to/datasets/logs

exec uv run python scripts/dataset_creation/build_benchmark_dataset.py \
    --config configs/dataset_creation/benchmarks_local.yaml
