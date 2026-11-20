#!/usr/bin/env bash
# GPU checks of the torch.compile fix for the pretraining encoder and decoder
# (branch compile_encoder_decoder). Plain bash, no SLURM; run it on a machine
# with one CUDA GPU (H200/B200 for timings that get reported).
#
#   REPO_ROOT=/path/to/Rem3Di bash scripts/gpu_checks/compile_encoder_decoder.sh
#
# Steps (each one runs even if an earlier one failed; the exit code is non-zero
# if any failed):
#   1. torch-repro: the PyTorch bug alone (nn.Linear on a pair tensor must fail
#      with CantSplit on torch 2.15, PairLinear must compile).
#   2. model-repro: the drug_like_small encoder and decoder compiled with dynamic
#      shapes, forward and backward over four batch shapes, one graph each.
#   3. pytest tests/model/test_compiled_encoder_decoder.py: compiled against eager
#      outputs and gradients (CPU and CUDA), one graph per module.
#   4./5. Pretraining smoke runs, compile on and compile off (same config,
#      400 training batches, 20 validation batches).
#   6. Summary table: per-window step_ms, preprocessor_ms, featurizer_ms,
#      recompiles, final training and validation losses.
#
# Environment (defaults in brackets):
#   REPO_ROOT        the repository checkout [the checkout this script is in]
#   OUTPUT_DIR       logs, configs and run directories
#                    [$HOME/remedi_compile_checks/<timestamp>]
#   DATASET_ROOT     directory holding pcqm4mv2/ and geom_drugs/ [the paths in
#                    configs/training/drug_like_small/training_config.yaml]
#   MACE_MODEL_PATH  MACE-POLAR-1-M.model [the path in the drug_like_small
#                    architecture config]
#   STAGE_DIRECTORY  copy the datasets to this local directory first [unset: read
#                    them in place]
#   TRAIN_BATCHES [400], VALIDATION_BATCHES [20]
#   UV_RUN           the command that runs Python in the repo's environment
#                    ["uv run"; on DAIS: "uv run --no-sync" with
#                    UV_PROJECT_ENVIRONMENT set]
#   SKIP_SMOKE=1     only steps 1-3
# torch inductor compiles C++ for the CPU tests: g++ >= 10 must be on PATH (or
# CXX set).
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/remedi_compile_checks/$(date +%Y%m%d_%H%M%S)}"
TRAIN_BATCHES="${TRAIN_BATCHES:-400}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-20}"
UV_RUN="${UV_RUN:-uv run}"
read -r -a UV_RUN_COMMAND <<< "${UV_RUN}"

if [ ! -f "${REPO_ROOT}/scripts/compile_checks.py" ]; then
    echo "REPO_ROOT=${REPO_ROOT} has no scripts/compile_checks.py (wrong checkout or branch?)" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/configs" "${OUTPUT_DIR}/runs"
cd "${REPO_ROOT}" || exit 2
# The MACE foundation models are full pickles.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "repository: ${REPO_ROOT} ($(git rev-parse --abbrev-ref HEAD 2>/dev/null) $(git rev-parse --short HEAD 2>/dev/null))"
echo "output:     ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || echo "nvidia-smi failed: no GPU?"

declare -a STEP_NAMES=()
declare -a STEP_RESULTS=()
record() {  # record <name> <exit code> <log>
    STEP_NAMES+=("$1")
    if [ "$2" -eq 0 ]; then STEP_RESULTS+=("ok"); else STEP_RESULTS+=("FAILED (exit $2, log $3)"); fi
}
run_step() {  # run_step <name> <command...>: output to the terminal and logs/<name>.log
    local name="$1"; shift
    local log="${OUTPUT_DIR}/logs/${name}.log"
    echo; echo "=== ${name}: $*"
    "$@" 2>&1 | tee "${log}"
    record "${name}" "${PIPESTATUS[0]}" "${log}"
}
python_in_repo() { "${UV_RUN_COMMAND[@]}" python "$@"; }

run_step torch_repro python_in_repo scripts/compile_checks.py torch-repro
run_step model_repro python_in_repo scripts/compile_checks.py model-repro
run_step compile_tests "${UV_RUN_COMMAND[@]}" pytest -q -p no:cacheprovider --durations=10 \
    packages/remedi/tests/model/test_compiled_encoder_decoder.py

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    CONFIG_ARGUMENTS=(--output-base "${OUTPUT_DIR}/runs" --configs-directory "${OUTPUT_DIR}/configs")
    [ -n "${DATASET_ROOT:-}" ] && CONFIG_ARGUMENTS+=(--dataset-root "${DATASET_ROOT}")
    [ -n "${MACE_MODEL_PATH:-}" ] && CONFIG_ARGUMENTS+=(--mace-model-path "${MACE_MODEL_PATH}")
    run_step smoke_configs python_in_repo scripts/compile_checks.py smoke-configs "${CONFIG_ARGUMENTS[@]}"

    SUMMARY_RUNS=()
    for variant in compile_on compile_off; do
        log="${OUTPUT_DIR}/logs/smoke_${variant}.log"
        # A frozen snapshot of the config in a fresh run directory; the last
        # line printed is the snapshot's training_config.yaml.
        snapshot="$(python_in_repo scripts/submit_job.py "${OUTPUT_DIR}/configs/${variant}.yaml" --no-submit 2>&1 | tail -1)"
        if [ ! -f "${snapshot}" ]; then
            echo "snapshot of ${variant} failed: ${snapshot}" | tee "${log}"
            record "smoke_${variant}" 1 "${log}"
            continue
        fi
        STAGE_ARGUMENTS=()
        [ -n "${STAGE_DIRECTORY:-}" ] && STAGE_ARGUMENTS=(--stage_directory "${STAGE_DIRECTORY}/${variant}")
        run_directory="$(dirname "${snapshot}")"
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
            --format=csv -l 10 > "${run_directory}/gpu_utilization.csv" 2>/dev/null &
        monitor=$!
        start=$(date +%s)
        run_step "smoke_${variant}" python_in_repo scripts/run_online_embedding_denoising_pretraining.py \
            --training_config "${snapshot}" "${STAGE_ARGUMENTS[@]}" \
            --max_train_batches "${TRAIN_BATCHES}" --max_val_batches "${VALIDATION_BATCHES}"
        echo "${variant}: $(( $(date +%s) - start )) s wall time" | tee -a "${log}"
        kill "${monitor}" 2>/dev/null
        SUMMARY_RUNS+=(--run "${variant}" "${run_directory}" "${OUTPUT_DIR}/logs/smoke_${variant}.log")
    done

    if [ "${#SUMMARY_RUNS[@]}" -gt 0 ]; then
        echo; echo "=== summary"
        python_in_repo scripts/compile_checks.py summarise "${SUMMARY_RUNS[@]}" \
            | tee "${OUTPUT_DIR}/summary.txt"
        echo "(baseline before the fix, compile off on H200, job 929582: step_ms per window"
        echo " 85/207/184/135/111/112/81/85, featurizer_ms 23/156/132/85/60/61/34/36;"
        echo " training loss 0.937, validation loss 0.861)"
    fi
fi

echo; echo "=== steps"
failures=0
for index in "${!STEP_NAMES[@]}"; do
    printf '  %-16s %s\n' "${STEP_NAMES[$index]}" "${STEP_RESULTS[$index]}"
    [ "${STEP_RESULTS[$index]}" = "ok" ] || failures=$((failures + 1))
done
echo "logs: ${OUTPUT_DIR}/logs"
exit "${failures}"
