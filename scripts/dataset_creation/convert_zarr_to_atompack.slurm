#!/bin/bash -l
# CPU-only: convert zarr datasets to structures.atp in place (one-off), then verify
# them. The arguments go to convert_zarr_to_atompack.py; add --dry-run --shadow-root DIR
# to change nothing:
#   sbatch scripts/dataset_creation/convert_zarr_to_atompack.slurm \
#       --report /dais/fs/scratch/wedigs/remedi/logs/conversion_pretraining.yaml \
#       /dais/fs/scratch/wedigs/remedi/pretraining
#SBATCH --job-name=convert_zarr_to_atompack
#SBATCH --account=mpo_inst
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/dais/fs/scratch/wedigs/remedi/logs/%x_%j.out
#SBATCH --chdir=/dais/u/wedigs/descriptors/Rem3Di
set -euo pipefail
.venv/bin/python scripts/dataset_creation/convert_zarr_to_atompack.py \
    --workers "${SLURM_CPUS_PER_TASK}" "$@"
