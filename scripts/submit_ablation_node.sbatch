#!/bin/bash -l
#SBATCH --job-name=ablation
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/training_runs/logs/slurm-%x-%j.out
#SBATCH --error=/path/to/training_runs/logs/slurm-%x-%j.err
#
# Pack up to four single-GPU pretraining runs onto one JUWELS Booster node
# (4x A100). JUWELS only allocates whole nodes, so an ablation of N configs is
# submitted as ceil(N/4) of these jobs by scripts/submit_job.py. Each config is
# pinned to its own GPU via CUDA_VISIBLE_DEVICES and runs as a background
# process; the job waits for all of them and fails if any single run fails.
#
# Not meant to be submitted by hand -- use:
#     uv run python scripts/submit_job.py cfgA.yaml cfgB.yaml cfgC.yaml cfgD.yaml
# which snapshots each config into a frozen training directory and calls sbatch
# on this script once per batch of four.

set -euo pipefail

PROJECT_DIR=/path/to/3DMolecularDescriptors
cd "${PROJECT_DIR}"
mkdir -p /path/to/training_runs/logs

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "Usage: sbatch submit_ablation_node.sbatch <cfg1> [cfg2] [cfg3] [cfg4]" >&2
    exit 1
fi

# The training runs read secrets (e.g. WANDB_API_KEY) from .env via
# `uv run --env-file`. Fail fast with a clear message if it's missing rather
# than letting each run die mid-startup.
if [ ! -f .env ]; then
    echo "Missing ${PROJECT_DIR}/.env (runs need e.g. WANDB_API_KEY)." >&2
    exit 1
fi

# 4 runs x 12 dataloader workers == 48 cores. Keep the math libraries
# single-threaded so the workers across the four runs do not oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Booster compute nodes have no internet (security policy). wandb online mode
# would block forever on wandb.init(); log offline and `wandb sync <run_dir>/wandb`
# from a login node afterwards.
export WANDB_MODE=offline

nvidia-smi --list-gpus

CONFIGS=("$@")

# Stage each run's dataset up front, sequentially, so the concurrently-launched
# runs never race on the same /dev/shm copy. setup_gpu_job.py is idempotent:
# configs sharing a dataset_path resolve to one staged copy.
declare -a STAGED
for CFG in "${CONFIGS[@]}"; do
    STAGED+=("$(uv run --env-file .env python scripts/setup_gpu_job.py "$CFG")")
done

# Launch one run per GPU as a background process.
declare -a PIDS
for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    RUN_DIR="$(dirname "$CFG")"
    echo "[gpu ${i}] config=${CFG} dataset=${STAGED[$i]}"
    # Colocate each run's offline wandb data with its checkpoints.
    CUDA_VISIBLE_DEVICES=${i} WANDB_DIR="${RUN_DIR}" uv run --env-file .env python \
        scripts/run_online_embedding_denoising_pretraining.py \
        --training_config "${CFG}" \
        --dataset_path "${STAGED[$i]}" \
        > "${RUN_DIR}/stdout.log" 2>&1 &
    PIDS[$i]=$!
done

# Wait for every run; fail the job if any single run failed.
rc=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[gpu ${i}] done: ${CONFIGS[$i]}"
    else
        echo "[gpu ${i}] FAILED: ${CONFIGS[$i]}" >&2
        rc=1
    fi
done
exit $rc
