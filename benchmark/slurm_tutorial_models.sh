#!/bin/bash
# The joint model of the tutorial (MOFA+ on RNA and morphology), on all cells; output goes to notebooks/cache.
#   cd benchmark && WORK_DIR=/path/to/work MORPH=../data/morph.h5ad sbatch slurm_tutorial_models.sh
#SBATCH --job-name=rcc-tutorial-models
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.log
##SBATCH --partition=...
##SBATCH --account=...
#SBATCH --gres=gpu:1
set -euo pipefail
: "${WORK_DIR:?set WORK_DIR to the directory written by prep.py (without --n-cells)}"
: "${MORPH:?set MORPH to the morphology features, data/morph.h5ad of the notebook}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run --project .. python train_tutorial_models.py "$WORK_DIR" "$MORPH" ../notebooks/cache
