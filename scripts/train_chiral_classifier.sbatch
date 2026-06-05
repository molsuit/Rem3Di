#!/bin/bash -l
#SBATCH --job-name=chiral_ablation
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/mace/wedig1/training_runs/logs/slurm-%x-%j.out
#SBATCH --error=/p/scratch/mace/wedig1/training_runs/logs/slurm-%x-%j.err
#
# Run the 2x2 chiral-classifier ablation (pseudoscalars x focal loss) packed
# onto one JUWELS Booster node (4x A100), one run per GPU. Each run trains the
# pseudoscalar preprocessor + encoder + classification head from scratch with a
# frozen MACE-POLAR featurizer (scripts/train_chiral_classifier.py), using the
# stored stratified split in the chiral_cat zarr. Checkpoints (best val balanced
# accuracy, REM3DI layout) land under each config's output_base.
#
# With no arguments it runs the four generated ablation configs; pass 1-4
# training_config.yaml paths to override. JUWELS allocates whole nodes, so this
# uses all four GPUs.
#
# Partition: defaults to `booster`. For a quick debug turnaround use the
# development partition `develbooster` (faster to schedule, but capped at
# ~2h walltime and a small node count) by overriding at submit time -- sbatch
# CLI flags take precedence over the #SBATCH directives below:
#     sbatch --partition=develbooster --time=02:00:00 \
#         scripts/train_chiral_classifier.sbatch
#
# Usage:
#     uv run python scripts/generate_chiral_ablation_configs.py   # once
#     sbatch scripts/train_chiral_classifier.sbatch
#     sbatch scripts/train_chiral_classifier.sbatch cfgA.yaml cfgB.yaml ...

set -euo pipefail

PROJECT_DIR=/p/project1/mace/wedig1/3DMolecularDescriptors
cd "${PROJECT_DIR}"
mkdir -p /p/scratch/mace/wedig1/training_runs/logs

# Default: the four 2x2 ablation runs (explicit order for a stable GPU mapping).
DEFAULT_CONFIGS=(
    configs/training/chiral_cat_ablation/ps_focal/training_config.yaml
    configs/training/chiral_cat_ablation/nops_focal/training_config.yaml
    configs/training/chiral_cat_ablation/ps_ce/training_config.yaml
    configs/training/chiral_cat_ablation/nops_ce/training_config.yaml
)
if [ "$#" -eq 0 ]; then
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
else
    CONFIGS=("$@")
fi
if [ "${#CONFIGS[@]}" -gt 4 ]; then
    echo "At most 4 configs per node (4x A100); got ${#CONFIGS[@]}." >&2
    exit 1
fi
for CFG in "${CONFIGS[@]}"; do
    if [ ! -f "${CFG}" ]; then
        echo "Training config not found: ${CFG}" >&2
        echo "Run: uv run python scripts/generate_chiral_ablation_configs.py" >&2
        exit 1
    fi
done

# MACE / e3nn checkpoints carry non-tensor globals (slice); torch>=2.6 defaults
# to weights_only=True and would refuse to load them.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# Four runs x 8 dataloader workers == 32 cores out of 48. Keep the math
# libraries single-threaded so workers across the four runs don't oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Booster compute nodes have no internet; wandb (if enabled in a config) must
# log offline and be synced from a login node afterwards. Harmless when wandb
# is disabled (the chiral configs set wandb_active: false).
export WANDB_MODE=offline

nvidia-smi --list-gpus

# Stage datasets up front, sequentially, so the concurrent runs never race on
# the same /dev/shm copy. setup_gpu_job.py is idempotent: the four runs share
# one dataset_path (chiral_cat) and so resolve to a single staged copy.
declare -a STAGED
for CFG in "${CONFIGS[@]}"; do
    STAGED+=("$(uv run python scripts/setup_gpu_job.py "${CFG}")")
done

# Launch one run per GPU as a background process.
declare -a PIDS
for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    RUN_DIR="$(dirname "${CFG}")"
    echo "[gpu ${i}] config=${CFG} dataset=${STAGED[$i]}"
    CUDA_VISIBLE_DEVICES=${i} uv run python \
        scripts/train_chiral_classifier.py \
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
