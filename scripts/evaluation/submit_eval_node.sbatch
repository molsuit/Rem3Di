#!/bin/bash -l
#SBATCH --job-name=eval
#SBATCH --account=mace
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/evaluation_results/logs/slurm-%x-%j.out
#SBATCH --error=/path/to/evaluation_results/logs/slurm-%x-%j.err
#
# Pack up to four single-GPU model evaluations onto one JUWELS Booster node
# (4x A100). JUWELS only allocates whole nodes, so evaluating N models is
# submitted as ceil(N/4) of these jobs by scripts/evaluation/submit_eval_jobs.py.
#
# Each argument is one *eval config directory* (as produced by
# generate_pcqm_novicreg_eval_configs.py) holding a unified `manifest.yaml`.
# Each is pinned to its own GPU via CUDA_VISIBLE_DEVICES and run as a background
# process executing the manifest (benchmark + retrieval, fault-tolerant) via
# run_eval.py; the job waits for all and fails if any single model eval fails.
#
# Not meant to be submitted by hand -- use:
#     uv run python scripts/evaluation/submit_eval_jobs.py \
#         configs/eval/pcqm_ablation_novicreg/*
# which calls sbatch on this script once per batch of four.

set -euo pipefail

PROJECT_DIR=/path/to/3DMolecularDescriptors
cd "${PROJECT_DIR}"
mkdir -p /path/to/evaluation_results/logs

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "Usage: sbatch submit_eval_node.sbatch <cfgdir1> [cfgdir2] [cfgdir3] [cfgdir4]" >&2
    exit 1
fi

# e3nn's wigner constants are pickled with `slice` objects, which the
# PyTorch >=2.6 `weights_only=True` default refuses. Force the legacy loader
# before any e3nn import happens.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# 4 evals share 48 cores -- keep the math libraries single-threaded so the
# learners / dataloaders across the four runs do not oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

nvidia-smi --list-gpus

CONFIG_DIRS=("$@")

# Launch one model eval per GPU as a background process running its unified
# manifest (benchmark + retrieval). The framework runner is itself fault
# tolerant — a failing task is recorded in status.yaml and the others still run
# — and run_eval.py exits non-zero if any task failed, so the node job surfaces
# it. Resource caches are keyed by model name, so the four concurrent processes
# never write the same cache file.
declare -a PIDS
for i in "${!CONFIG_DIRS[@]}"; do
    CFG_DIR="${CONFIG_DIRS[$i]}"
    NAME="$(basename "${CFG_DIR}")"
    LOG="/path/to/evaluation_results/logs/eval-${NAME}.log"
    echo "[gpu ${i}] model=${NAME} config_dir=${CFG_DIR}"
    CUDA_VISIBLE_DEVICES=${i} uv run python scripts/evaluation/run_eval.py \
        --config "${CFG_DIR}/manifest.yaml" > "${LOG}" 2>&1 &
    PIDS[$i]=$!
done

# Wait for every eval; fail the job if any single one failed.
rc=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[gpu ${i}] done: ${CONFIG_DIRS[$i]}"
    else
        echo "[gpu ${i}] FAILED: ${CONFIG_DIRS[$i]}" >&2
        rc=1
    fi
done
exit $rc
