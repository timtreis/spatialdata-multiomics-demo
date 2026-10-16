#!/bin/bash
# One array task per (method, fold): 7 methods x 3 folds = 21 tasks.
#   cd benchmark && WORK_DIR=/path/to/work sbatch slurm_benchmark.sh
# Adapt the partition/account (and a GPU for scvi/totalvi, e.g. --gres=gpu:1) to your cluster.
#SBATCH --job-name=rcc-benchmark
#SBATCH --array=0-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.log
##SBATCH --partition=...
##SBATCH --account=...
set -euo pipefail
: "${WORK_DIR:?set WORK_DIR to the directory written by prep.py}"
METHODS=(rna_pca prot_markers concat scvi totalvi wnn mofa)
method=${METHODS[$((SLURM_ARRAY_TASK_ID % ${#METHODS[@]}))]}
fold=$((SLURM_ARRAY_TASK_ID / ${#METHODS[@]}))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run --project .. python methods.py "$method" "$fold" "$WORK_DIR"
