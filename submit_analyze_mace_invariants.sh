#!/bin/bash
#SBATCH --job-name=analyze_mace_invariants
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gpus=1
#SBATCH --time=2:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate

nvidia-smi --list-gpus

srun uv run --env-file .env scripts/evaluation/analyze_mace_invariants.py \
    --config scripts/evaluation/analyze_mace_invariants.yaml
