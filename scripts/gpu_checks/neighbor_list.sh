#!/usr/bin/env bash
# GPU checks of the neighbour-list MACE featurizer (branch compiled_featurizer), on any
# machine with one CUDA GPU; no SLURM. Run it from anywhere; it works in the
# repository root given by --repo (default: the checkout this script is in).
#
#   1. GPU tests: the featurizer tests incl. cuequivariance + torch.compile on CUDA and
#      MACE-POLAR-1-M exactness (test_mace_featurizer.py, test_assembled_model.py,
#      test_featurizer_gpu_checks.py).
#   2. Benchmark: MACE-POLAR-1-M node features per ~5250-atom batch, float32 + cueq,
#      small (PCQM4Mv2) and large (GEOM-drugs, 60-120 atoms) molecule batches, dense
#      graph (every intra-molecule pair, the featurizer before this branch) against the
#      neighbour list, eager and compiled: ms, peak memory, max feature difference.
#   3. Pretraining smoke run, compile off, 400 training and 20 validation batches
#      (configs/training/drug_like_small with the paths below): throughput windows
#      next to the dense baseline, and the recompiles logged.
#   Then a summary table. Every step runs even if an earlier one failed.
#
# Options (each defaults to the environment variable named in brackets, then to the
# DAIS path):
#   --repo DIR          repository root                     [REPO_ROOT]
#   --model_path FILE   MACE-POLAR-1-M.model                [MACE_POLAR_MODEL_PATH]
#   --pcqm4mv2 DIR      pretraining dataset pcqm4mv2        [PCQM4MV2_PATH]
#                       (table.parquet, structures.atp, selections/diverse_100k)
#   --geom_drugs DIR    pretraining dataset geom_drugs      [GEOM_DRUGS_PATH]
#                       (table.parquet, structures.atp, selections/top1)
#   --output_dir DIR    benchmark files, smoke configs, runs, logs
#                       [OUTPUT_DIR, default ${TMPDIR:-/tmp}/neighbor_list_gpu_checks/<time>]
#   --num_workers N     data-loader workers of the smoke run [NUM_WORKERS, 12]
#   --stage_directory DIR  copy the datasets there first (node-local disk) [STAGE_DIRECTORY]
#   --skip STEPS        comma list of tests,benchmark,smoke to skip
# Environment:
#   UV_RUN              how to run Python (default "uv run"; on DAIS's shared venv use
#                       UV_PROJECT_ENVIRONMENT=... UV_RUN="uv run --no-sync")
#   CXX                 a C++ compiler >= gcc 10 for torch inductor's CPU code, if the
#                       system one is older
# The MACE-OFF23 medium tests read ~/.cache/mace/MACE-OFF23_medium.model; it is
# downloaded first if missing (they are skipped if that fails).
#
# Example:
#   bash scripts/gpu_checks/neighbor_list.sh --repo ~/Rem3Di --model_path ~/models/MACE-POLAR-1-M.model \
#       --pcqm4mv2 /data/remedi/pretraining/pcqm4mv2 --geom_drugs /data/remedi/pretraining/geom_drugs
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_PATH="${MACE_POLAR_MODEL_PATH:-/dais/u/wedigs/graph_electrostatics/MACE-POLAR-1-M.model}"
PCQM4MV2="${PCQM4MV2_PATH:-/dais/fs/scratch/wedigs/remedi/pretraining/pcqm4mv2}"
GEOM_DRUGS="${GEOM_DRUGS_PATH:-/dais/fs/scratch/wedigs/remedi/pretraining/geom_drugs}"
OUTPUT_DIR="${OUTPUT_DIR:-${TMPDIR:-/tmp}/neighbor_list_gpu_checks/$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-12}"
STAGE_DIRECTORY="${STAGE_DIRECTORY:-}"
SKIP=""
UV_RUN="${UV_RUN:-uv run}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo) REPO_ROOT="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --pcqm4mv2) PCQM4MV2="$2"; shift 2 ;;
        --geom_drugs) GEOM_DRUGS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --stage_directory) STAGE_DIRECTORY="$2"; shift 2 ;;
        --skip) SKIP="$2"; shift 2 ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    esac
done

for path in "$REPO_ROOT/scripts/featurizer_gpu_checks.py" "$MODEL_PATH" "$PCQM4MV2" "$GEOM_DRUGS"; do
    if [ ! -e "$path" ]; then
        echo "missing: $path (see --help)" >&2
        exit 2
    fi
done
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
PCQM4MV2="$(cd "$PCQM4MV2" && pwd)"
GEOM_DRUGS="$(cd "$GEOM_DRUGS" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
cd "$REPO_ROOT"

# The MACE foundation models are full pickles.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export RUN_POLAR_TESTS=1
export MACE_POLAR_MODEL_PATH="$MODEL_PATH"
read -r -a UV <<< "$UV_RUN"
CHECKS=(scripts/featurizer_gpu_checks.py)
declare -A STATUS

skipped() { [[ ",$SKIP," == *",$1,"* ]]; }
log() { echo "[run_gpu_checks $(date +%H:%M:%S)] $*"; }

log "repository $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
log "output $OUTPUT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
"${UV[@]}" python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print('torch', torch.__version__, torch.cuda.get_device_name())" \
    || { echo "PyTorch sees no CUDA device" >&2; exit 1; }

# ---------------------------------------------------------------- 1. GPU tests
if skipped tests; then
    STATUS[tests]="skipped"
else
    if [ ! -e "$HOME/.cache/mace/MACE-OFF23_medium.model" ]; then
        log "downloading MACE-OFF23 medium for the tests"
        "${UV[@]}" python -c "from mace.calculators.foundations_models import mace_off; mace_off(model='medium', device='cpu')" \
            > "$OUTPUT_DIR/mace_off_download.log" 2>&1 \
            || log "download failed: the MACE-OFF tests will be skipped"
    fi
    log "1/3 GPU tests"
    "${UV[@]}" python -m pytest \
        packages/remedi/tests/model/test_mace_featurizer.py \
        packages/remedi/tests/model/test_assembled_model.py \
        packages/remedi/tests/test_featurizer_gpu_checks.py \
        -p no:cacheprovider -W ignore -rs 2>&1 | tee "$OUTPUT_DIR/tests.log"
    TEST_EXIT=${PIPESTATUS[0]}
    TEST_LINE="$(grep -E '(passed|failed|error)' "$OUTPUT_DIR/tests.log" | tail -1 | tr -d '=' | xargs)"
    STATUS[tests]="exit $TEST_EXIT: $TEST_LINE"
fi

# ---------------------------------------------------------------- 2. benchmark
BENCHMARK_DIR="$OUTPUT_DIR/benchmark"
if skipped benchmark; then
    STATUS[benchmark]="skipped"
else
    log "2/3 benchmark (each mode in its own process)"
    mkdir -p "$BENCHMARK_DIR"
    FAILED_MODES=""
    for MODE in neighbor_eager neighbor_compiled dense_eager dense_compiled; do
        log "  $MODE"
        "${UV[@]}" python "${CHECKS[@]}" benchmark --mode "$MODE" \
            --model_path "$MODEL_PATH" --pcqm4mv2 "$PCQM4MV2" --geom_drugs "$GEOM_DRUGS" \
            --output_dir "$BENCHMARK_DIR" > "$BENCHMARK_DIR/$MODE.log" 2>&1 \
            || { FAILED_MODES="$FAILED_MODES $MODE"; tail -20 "$BENCHMARK_DIR/$MODE.log"; }
    done
    STATUS[benchmark]="${FAILED_MODES:+failed:$FAILED_MODES}"
    STATUS[benchmark]="${STATUS[benchmark]:-ok}"
fi

# ---------------------------------------------------------------- 3. smoke run
RUN_DIRECTORY=""
if skipped smoke; then
    STATUS[smoke]="skipped"
else
    log "3/3 pretraining smoke run (compile off, 400 + 20 batches)"
    SMOKE_LOG="$OUTPUT_DIR/smoke.log"
    CONFIG="$("${UV[@]}" python "${CHECKS[@]}" smoke-config \
        --model_path "$MODEL_PATH" --pcqm4mv2 "$PCQM4MV2" --geom_drugs "$GEOM_DRUGS" \
        --output_base "$OUTPUT_DIR/runs" --config_dir "$OUTPUT_DIR/smoke_config" \
        --num_workers "$NUM_WORKERS" 2> "$OUTPUT_DIR/smoke_config.log" | tail -1)"
    SNAPSHOT=""
    if [ -n "$CONFIG" ] && [ -f "$CONFIG" ]; then
        SNAPSHOT="$("${UV[@]}" python scripts/submit_job.py "$CONFIG" --no-submit 2>> "$OUTPUT_DIR/smoke_config.log" | tail -1)"
    fi
    if [ -z "$SNAPSHOT" ] || [ ! -f "$SNAPSHOT" ]; then
        STATUS[smoke]="config failed (see $OUTPUT_DIR/smoke_config.log)"
        tail -20 "$OUTPUT_DIR/smoke_config.log"
    else
        RUN_DIRECTORY="$(dirname "$SNAPSHOT")"
        STAGE_ARGUMENTS=()
        if [ -n "$STAGE_DIRECTORY" ]; then
            STAGE_ARGUMENTS=(--stage_directory "$STAGE_DIRECTORY")
        fi
        # Recompiles (if anything compiles) are logged, to be counted below.
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_LOGS="recompiles" \
            "${UV[@]}" python scripts/run_online_embedding_denoising_pretraining.py \
            --training_config "$SNAPSHOT" "${STAGE_ARGUMENTS[@]}" \
            --max_train_batches 400 --max_val_batches 20 > "$SMOKE_LOG" 2>&1
        SMOKE_EXIT=$?
        RECOMPILES="$(grep -ci 'recompil' "$SMOKE_LOG")"
        STATUS[smoke]="exit $SMOKE_EXIT, $RECOMPILES recompile log lines, run $RUN_DIRECTORY"
        [ "$SMOKE_EXIT" -eq 0 ] || tail -30 "$SMOKE_LOG"
    fi
fi

# ---------------------------------------------------------------- summary
echo
echo "================================ summary ================================"
SUMMARY_ARGUMENTS=()
skipped benchmark || SUMMARY_ARGUMENTS+=(--benchmark_dir "$BENCHMARK_DIR")
[ -n "$RUN_DIRECTORY" ] && SUMMARY_ARGUMENTS+=(--run_directory "$RUN_DIRECTORY")
if [ "${#SUMMARY_ARGUMENTS[@]}" -gt 0 ]; then
    "${UV[@]}" python "${CHECKS[@]}" summary "${SUMMARY_ARGUMENTS[@]}" | tee "$OUTPUT_DIR/summary.txt"
fi
echo
for STEP in tests benchmark smoke; do
    printf '%-10s %s\n' "$STEP" "${STATUS[$STEP]}" | tee -a "$OUTPUT_DIR/summary.txt"
done
echo "everything is in $OUTPUT_DIR"
