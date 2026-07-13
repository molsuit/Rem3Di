#!/bin/bash -l
#SBATCH --job-name=nvmolkit_smoke
#SBATCH --account=mace
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/datasets/logs/slurm-%j.out
#SBATCH --error=/path/to/datasets/logs/slurm-%j.err

set -euo pipefail

cd /path/to/3DMolecularDescriptors
exec uv run python scripts/dataset_creation/nvmolkit_smoke.py
