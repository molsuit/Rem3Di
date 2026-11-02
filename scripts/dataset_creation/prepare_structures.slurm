#!/bin/bash -l
# CPU-only: prepare the structure corpora of configs/dataset_creation/prepare_structures.yaml.
# Restartable: resubmitting after a timeout redoes only unfinished shards.
#   sbatch scripts/dataset_creation/prepare_structures.slurm [--only tmqm ...]
#SBATCH --job-name=prepare_structures
#SBATCH --account=mpo_inst
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --output=/dais/fs/scratch/wedigs/remedi/logs/%x_%j.out
#SBATCH --chdir=/dais/u/wedigs/descriptors/Rem3Di
set -euo pipefail
uv run --no-sync python scripts/dataset_creation/prepare_structures.py \
    --config configs/dataset_creation/prepare_structures.yaml "$@"
