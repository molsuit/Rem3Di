#!/bin/bash -l
#SBATCH --job-name=pcqm_subset
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/path/to/datasets/logs/slurm-%x-%j.out
#SBATCH --error=/path/to/datasets/logs/slurm-%x-%j.err
#
# Build a fixed-size pcqm4m structure subset on /p/scratch for fast pretraining
# iteration. CPU-only (SDF parse + filter + copy to zarr; embeddings are
# computed online during training). The first positional argument is the subset
# size; it defaults to 500000.
#
# Submit with:
#     sbatch scripts/dataset_creation/build_pcqm_subset_juwels.sbatch [N_STRUCTURES]

set -euo pipefail

cd /path/to/3DMolecularDescriptors
mkdir -p /path/to/datasets/logs

N_STRUCTURES="${1:-500000}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Building pcqm subset of ${N_STRUCTURES} structures"

exec uv run python scripts/dataset_creation/load_pcqm100k_juwels.py \
    --n-structures "${N_STRUCTURES}"
