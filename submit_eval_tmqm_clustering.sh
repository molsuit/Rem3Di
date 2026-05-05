#!/bin/bash
#SBATCH --job-name=eval_tmqm_clustering
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=1:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate

nvidia-smi --list-gpus

MODEL_DIR="$1"

srun uv run --env-file .env scripts/evaluation/eval_tmqm_clustering.py \
    --model-dir "${MODEL_DIR}"
