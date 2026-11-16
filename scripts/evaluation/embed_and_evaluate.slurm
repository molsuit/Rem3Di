#!/bin/bash -l
#SBATCH --job-name=embed_and_evaluate
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --chdir=/dais/u/wedigs/descriptors/Rem3Di
#SBATCH --output=/dais/fs/scratch/wedigs/remedi/logs/%x_%j.out
#
# Embed every dataset a manifest needs with one Rem3Di checkpoint, then run the
# manifest (benchmark panels, descriptor analyses) on one H200 or B200:
#     sbatch --time=00:45:00 scripts/evaluation/embed_and_evaluate.slurm <config.yaml>
# Pass a --time that fits the run (see configs/evaluation/descriptor_analysis/).
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch scripts/evaluation/embed_and_evaluate.slurm <config.yaml>" >&2
    exit 1
fi

module load gcc/15
export CXX="$(which g++)"
# The CPU half (UMAP, HDBSCAN, learners) uses every allocated core.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# The MACE foundation models are full pickles.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
uv run --no-sync python scripts/evaluation/embed_and_evaluate.py --config "$1"
