#!/bin/bash -l
#SBATCH --job-name=comparison_geom
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/datasets/logs/slurm-%j.out
#SBATCH --error=/path/to/datasets/logs/slurm-%j.err
#
# DatasetComparison with GEOM drugs as the sole pretrain corpus.
#
# Submit with:
#     sbatch scripts/dataset_creation/run_dataset_comparison_geom_only.sbatch

set -euo pipefail

cd /path/to/3DMolecularDescriptors
mkdir -p /path/to/datasets/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

exec uv run python scripts/dataset_creation/run_dataset_comparison.py \
    --config configs/dataset_creation/dataset_comparison_geom_only.yaml
