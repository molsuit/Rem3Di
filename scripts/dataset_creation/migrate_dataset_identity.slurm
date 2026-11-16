#!/bin/bash -l
# CPU-only: migrate datasets to the current identity and label layout (one-off),
# then verify them. The arguments go to migrate_dataset_identity.py; add --dry-run
# to change nothing:
#   sbatch scripts/dataset_creation/migrate_dataset_identity.slurm --dry-run \
#       /dais/fs/scratch/wedigs/remedi/pretraining
#SBATCH --job-name=migrate_dataset_identity
#SBATCH --account=mpo_inst
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/dais/fs/scratch/wedigs/remedi/logs/%x_%j.out
#SBATCH --chdir=/dais/u/wedigs/descriptors/Rem3Di
set -euo pipefail
.venv/bin/python scripts/dataset_creation/migrate_dataset_identity.py \
    --workers "${SLURM_CPUS_PER_TASK}" "$@"
