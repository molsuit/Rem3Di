#!/bin/bash -l
#SBATCH --job-name=pretrain
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --chdir=/dais/u/wedigs/descriptors/Rem3Di
#SBATCH --output=/dais/fs/scratch/wedigs/remedi/logs/%x_%j.out
#
# One pretraining run on one H200 or B200 (whichever is free). Submit through
#     uv run python scripts/submit_job.py cfgA.yaml cfgB.yaml ...
# which snapshots each config into its training directory and submits one job per
# config. Arguments after the config (e.g. --max_train_batches 20) are passed on
# to the training script.
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch scripts/pretrain.slurm <training_config.yaml> [script options]" >&2
    exit 1
fi
CONFIG="$1"
shift

# torch.compile builds C++ with the compiler it finds; the system gcc 7.5 is too old.
module load gcc/15
export CXX="$(which g++)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# The MACE foundation models are full pickles.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Secrets such as WANDB_API_KEY come from .env when it exists.
ENV_FILE=()
if [ -f .env ]; then
    ENV_FILE=(--env-file .env)
fi

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
# Keep wandb's files next to the run's checkpoints.
WANDB_DIR="$(dirname "${CONFIG}")" uv run --no-sync "${ENV_FILE[@]}" python \
    scripts/run_online_embedding_denoising_pretraining.py \
    --training_config "${CONFIG}" "$@"
