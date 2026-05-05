#!/bin/bash
#SBATCH --job-name=umap_grid_sweep
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=4:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate

nvidia-smi --list-gpus

DESCRIPTORS_PATH="$1"
OUTPUT_DIR="$2"

srun uv run --env-file .env scripts/evaluation/umap_grid_sweep.py \
    --descriptors-path "${DESCRIPTORS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --save-coords
