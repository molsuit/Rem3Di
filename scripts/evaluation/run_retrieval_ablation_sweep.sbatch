#!/bin/bash -l
#SBATCH --job-name=retrieval_sweep
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-10
#SBATCH --output=/p/scratch/mace/wedig1/evaluation_results/logs/slurm-%A_%a.out
#SBATCH --error=/p/scratch/mace/wedig1/evaluation_results/logs/slurm-%A_%a.err
#
# Full-100k vector-retrieval eval across every run in the pcqm_ablation dir.
# One array task per model (11 models, run in parallel). Each task embeds the
# whole pcqm100k dataset with its checkpoint and runs Task A (Tanimoto, 500
# queries) + Task C (nearest-molecule, 50 sampled SMILES).
#
# Submit with:
#     sbatch scripts/evaluation/run_retrieval_ablation_sweep.sbatch

set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /p/project1/mace/wedig1/3DMolecularDescriptors
mkdir -p /p/scratch/mace/wedig1/evaluation_results/logs

ABLATION=/p/scratch/mace/wedig1/training_runs/pcqm_ablation
DATASET=/p/scratch/mace/wedig1/datasets/pcqm4m/pcqm100k
MACE_MODEL=/p/project1/mace/wedig1/3DMolecularDescriptors/MACE-POLAR-1-M.model
OUTBASE=/p/scratch/mace/wedig1/evaluation_results/retrieval/pcqm100k_ablation

# One entry per array index (order is fixed so re-runs are reproducible).
RUNS=(
    22-2026_05_19_12_56_55-pcqm_agg_mean
    23-2026_05_19_12_57_12-pcqm_agg_pma
    24-2026_05_19_12_57_28-pcqm_baseline
    25-2026_05_19_12_57_45-pcqm_depth_2
    26-2026_05_19_12_58_12-pcqm_depth_6
    27-2026_05_19_12_58_37-pcqm_dim_128
    28-2026_05_19_12_59_02-pcqm_dim_32
    29-2026_05_19_12_59_16-pcqm_dropout_0.1
    30-2026_05_19_12_59_36-pcqm_lr_1e-3
    31-2026_05_19_12_59_48-pcqm_lr_1e-4
    32-2026_05_19_13_13_33-pcqm_dropout_0.5
)

RUN=${RUNS[${SLURM_ARRAY_TASK_ID}]}
MODEL_DIR=${ABLATION}/${RUN}
# Clean model name (strip the "NN-DATE-" prefix) for cache keys / output dir.
NAME=$(echo "${RUN}" | sed 's/^[0-9]*-[0-9_]*-//')
OUTPUT_DIR=${OUTBASE}/${NAME}

echo "[array ${SLURM_ARRAY_TASK_ID}] model=${NAME} dir=${MODEL_DIR}"

exec uv run python scripts/evaluation/run_retrieval_eval.py \
    --model-dir "${MODEL_DIR}" \
    --model-name "${NAME}" \
    --mace-model-path "${MACE_MODEL}" \
    --dataset "${DATASET}" \
    --dataset-id pcqm100k \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda \
    --batch-size 64 \
    --k 10 \
    --n-query-sample 500 \
    --n-global-pairs 100000 \
    --n-query-smiles 50 \
    --query-sample-seed 0
