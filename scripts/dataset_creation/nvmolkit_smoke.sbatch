#!/bin/bash -l
#SBATCH --job-name=nvmolkit_smoke
#SBATCH --account=mace
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/p/scratch/mace/wedig1/datasets/logs/slurm-%j.out
#SBATCH --error=/p/scratch/mace/wedig1/datasets/logs/slurm-%j.err

set -euo pipefail

cd /p/project1/mace/wedig1/3DMolecularDescriptors
exec uv run python scripts/dataset_creation/nvmolkit_smoke.py
