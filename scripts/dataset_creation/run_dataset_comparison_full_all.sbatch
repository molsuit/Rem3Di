#!/bin/bash -l
#SBATCH --job-name=comparison_full_all
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/datasets/logs/slurm-%j.out
#SBATCH --error=/path/to/datasets/logs/slurm-%j.err
#
# DatasetComparison: 3.5M pcqm + 290k GEOM drugs as pretrain, every zarr in
# the benchmark directory (37 panels) as eval. Larger budget than the 4-panel
# variant because the NN-Tanimoto scales linearly in eval molecule count.
#
# Submit with:
#     sbatch scripts/dataset_creation/run_dataset_comparison_full_all.sbatch

set -euo pipefail

cd /path/to/3DMolecularDescriptors
mkdir -p /path/to/datasets/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

exec uv run python scripts/dataset_creation/run_dataset_comparison.py \
    --config configs/dataset_creation/dataset_comparison_full_all.yaml
