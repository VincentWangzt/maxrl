#!/usr/bin/env bash
set -euo pipefail

# All experiment settings are explicit here; no ambient experiment overrides.
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
OUTPUT_DIR="${DATA_DIR}"
TRAIN_COUNT=10000000
EVAL_COUNT=1024
DIMENSION=2
OBSERVATIONS=64
SIGMA=0.1 # Same standard deviation for independent context and query noise.
CAPACITY=1024
GPU_ID="" # Dataset generation is CPU-only.
CPU_THREADS=4

source "${VENV_DIR}/bin/activate"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}/noisy-regression:${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
cd "${REPO_ROOT}"
exec python -m noisy_regression.data --output "${OUTPUT_DIR}" \
  --train-count "${TRAIN_COUNT}" --eval-count "${EVAL_COUNT}" \
  --dimension "${DIMENSION}" --observations "${OBSERVATIONS}" \
  --sigma "${SIGMA}" --capacity "${CAPACITY}"
