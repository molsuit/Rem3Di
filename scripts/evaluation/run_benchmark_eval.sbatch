#!/usr/bin/env bash
#SBATCH --job-name=benchmark_eval
#SBATCH --account=brics.s5f
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/s5f/wedigs.s5f/evaluation_results/logs/slurm-%j.out
#SBATCH --error=/scratch/s5f/wedigs.s5f/evaluation_results/logs/slurm-%j.err
#
# Runs the descriptor-probe benchmark eval (descriptors x learners cross-
# product) over every prepared zarr in the config's eval_root and writes
# results.csv + results.yaml under output_dir.
#
# Submit with:
#     sbatch scripts/evaluation/run_benchmark_eval.sbatch

set -euo pipefail

# e3nn's wigner constants are pickled with `slice` objects, which the
# PyTorch >=2.6 `weights_only=True` default refuses. Force the legacy
# loader before any e3nn import happens.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd /home/s5f/wedigs.s5f/3DMolecularDescriptors
mkdir -p /scratch/s5f/wedigs.s5f/evaluation_results/logs

exec uv run python scripts/evaluation/run_benchmark_eval.py \
    --config configs/eval/benchmark_eval_local.yaml
