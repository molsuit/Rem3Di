#!/bin/bash -l
#SBATCH --job-name=comparison_full
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/mace/wedig1/datasets/logs/slurm-%j.out
#SBATCH --error=/p/scratch/mace/wedig1/datasets/logs/slurm-%j.err
#
# Full DatasetComparison: 3.5M pcqm + 290k GEOM drugs as pretrain,
# four MoleculeNet benchmarks as eval. Wall time budget is generous;
# the BitBIRCH fit on ~3.8M packed FPs is the dominant cost.
#
# Submit with:
#     sbatch scripts/dataset_creation/run_dataset_comparison_full.sbatch

set -euo pipefail

cd /p/project1/mace/wedig1/3DMolecularDescriptors
mkdir -p /p/scratch/mace/wedig1/datasets/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

exec uv run python scripts/dataset_creation/run_dataset_comparison.py \
    --config configs/dataset_creation/dataset_comparison_full.yaml
