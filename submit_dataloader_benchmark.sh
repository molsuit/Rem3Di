#!/bin/bash
#SBATCH --job-name=dl-bench
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate

nvidia-smi --list-gpus

BENCHMARK_CONFIG="${1:?Usage: sbatch submit_dataloader_benchmark.sh <benchmark_config_path>}"

LOCAL_DATASET_PATH=$(uv run --env-file .env scripts/setup_gpu_job.py "$BENCHMARK_CONFIG")

srun uv run --env-file .env scripts/profiling/profile_dataloader.py \
    --config "$BENCHMARK_CONFIG" \
    --dataset_path "$LOCAL_DATASET_PATH"
