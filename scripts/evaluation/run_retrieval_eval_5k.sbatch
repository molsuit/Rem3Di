#!/bin/bash -l
#SBATCH --job-name=retrieval_5k
#SBATCH --account=mace
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/mace/wedig1/evaluation_results/logs/slurm-%j.out
#SBATCH --error=/p/scratch/mace/wedig1/evaluation_results/logs/slurm-%j.err
#
# Smoke test of the vector-retrieval eval on a GPU node: embeds the first 5k
# structures of pcqm100k with the pcqm_ablation baseline checkpoint, then runs
# the Tanimoto-similarity (Task A) and nearest-molecule (Task C) tasks.
#
# --max-structures 5000 limits BOTH the embedding compute and the store, so
# this is genuinely cheap. Drop the flag (and bump --time / use --partition
# booster) for the full 100k run.
#
# Submit with:
#     sbatch scripts/evaluation/run_retrieval_eval_5k.sbatch

set -euo pipefail

# e3nn's wigner constants are pickled with `slice` objects, which the
# PyTorch >=2.6 `weights_only=True` default refuses. Force the legacy loader
# before any e3nn import happens.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /p/project1/mace/wedig1/3DMolecularDescriptors
mkdir -p /p/scratch/mace/wedig1/evaluation_results/logs

MODEL_DIR=/p/scratch/mace/wedig1/training_runs/pcqm_ablation/24-2026_05_19_12_57_28-pcqm_baseline
DATASET=/p/scratch/mace/wedig1/datasets/pcqm4m/pcqm100k
MACE_MODEL=/p/project1/mace/wedig1/3DMolecularDescriptors/MACE-POLAR-1-M.model
OUTPUT_DIR=/p/scratch/mace/wedig1/evaluation_results/retrieval/pcqm100k_baseline_5k

exec uv run python scripts/evaluation/run_retrieval_eval.py \
    --model-dir "${MODEL_DIR}" \
    --model-name pcqm_baseline \
    --mace-model-path "${MACE_MODEL}" \
    --dataset "${DATASET}" \
    --dataset-id pcqm100k \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --batch-size 64 \
    --max-structures 5000 \
    --k 10 \
    --n-query-sample 200 \
    --n-global-pairs 20000 \
    --n-query-smiles 20 \
    --query-sample-seed 0
