#!/bin/bash -l
#SBATCH --job-name=time_butina
#SBATCH --account=mpo_inst
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=daisg301,daisg302
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --chdir=/dais/u/wedigs/descriptors/Rem3Di
#SBATCH --output=/dais/fs/scratch/wedigs/remedi/logs/%x_%j.out

# Butina timing study on nested PCQM4Mv2 samples (BENCHMARK_DATA_FORMAT.md §5.2.3).
set -euo pipefail
# nvmolkit 0.6 needs libcudart >= 12.8 (cudaStreamGetDevice); torch's cu126 wheels pin
# nvidia-cuda-runtime-cu12 12.6, so preload the cluster's 12.8 runtime
# (BENCHMARK_REFACTOR_ISSUES.md).
export LD_PRELOAD=/mpcdf/soft/SLE_15/packages/x86_64/cuda/12.8.1/lib64/libcudart.so.12
uv run --no-sync python scripts/dataset_creation/time_butina.py \
    --dataset /dais/fs/scratch/wedigs/remedi/pretraining/pcqm4mv2 \
    --sizes 100000 300000 1000000 3500000 \
    --coarse-cutoffs 0.65 0.6 0.7 0.8 \
    --fine-cutoff 0.4 \
    --budget-seconds 14400 \
    --output /dais/fs/scratch/wedigs/remedi/logs/butina_timing_${SLURM_JOB_ID}.yaml
