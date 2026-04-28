#!/bin/bash
#SBATCH --job-name=3dscriptors
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=10:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate

nvidia-smi --list-gpus

TRAINING_CONFIG="${1:?Usage: sbatch submit_training_run.sh <training_config_path>}"

LOCAL_DATASET_PATH=$(uv run --env-file .env scripts/setup_gpu_job.py "$TRAINING_CONFIG")

srun uv run --env-file .env scripts/run_online_embedding_denoising_pretraining.py \
    --training_config "$TRAINING_CONFIG" \
    --dataset_path "$LOCAL_DATASET_PATH"
