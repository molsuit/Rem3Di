#!/bin/bash -l
#SBATCH --job-name=benchmark_build
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/mace/wedig1/datasets/logs/slurm-%j.out
#SBATCH --error=/p/scratch/mace/wedig1/datasets/logs/slurm-%j.err
#
# JUWELS Cluster build of the full benchmark panel (every registered
# MoleculeNet + TDC ADMET + Polaris dataset). Re-running is idempotent:
# zarrs whose directory already exists with a manifest are skipped.
#
# Submit with:
#     sbatch scripts/dataset_creation/build_benchmark_dataset_juwels.sbatch

set -euo pipefail

cd /p/project1/mace/wedig1/3DMolecularDescriptors
mkdir -p /p/scratch/mace/wedig1/datasets/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

exec uv run python scripts/dataset_creation/build_benchmark_dataset.py \
    --config configs/dataset_creation/benchmarks_local.yaml
