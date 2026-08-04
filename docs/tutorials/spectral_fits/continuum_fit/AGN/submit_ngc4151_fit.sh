#!/bin/bash

#SBATCH --job-name=ngc4151-cpl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=a100:1
#SBATCH --output=NGC4151-cpl.out
#SBATCH --error=NGC4151-cpl.err
#SBATCH --open-mode=truncate

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export COSI_DEVICE=cuda:0

# The normalizing-flow workers may require a higher file-descriptor limit.
ulimit -n 65535 || true

ANALYSIS_DIR=/home/parshap/cosipy/docs/tutorials/spectral_fits/continuum_fit/AGN
NOTEBOOK=NGC4151_fit_unbinned.ipynb
EXECUTED_NOTEBOOK=NGC4151_fit_unbinned_executed.ipynb
CONDA=/home/parshap/miniforge3/bin/conda

cd "$ANALYSIS_DIR"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

srun "$CONDA" run \
    --no-capture-output \
    -n cosipy \
    jupyter nbconvert \
    --to notebook \
    --execute "$NOTEBOOK" \
    --output "$EXECUTED_NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1

echo "Completed: $EXECUTED_NOTEBOOK"
