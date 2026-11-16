#!/bin/bash -l
# CPU-only: write the selections of configs/dataset_creation/select_structures.yaml.
# Unchanged selections are kept, so a rerun only redoes what changed.
#   sbatch scripts/dataset_creation/select_structures.slurm
#SBATCH --job-name=select_structures
#SBATCH --account=mpo_inst
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/dais/fs/scratch/wedigs/remedi/logs/%x_%j.out
#SBATCH --chdir=/dais/u/wedigs/descriptors/Rem3Di
set -euo pipefail
uv run --no-sync python scripts/dataset_creation/select_structures.py \
    --config configs/dataset_creation/select_structures.yaml
