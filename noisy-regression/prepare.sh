#!/usr/bin/env bash
set -euo pipefail

# All experiment settings are explicit here; no ambient experiment overrides.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
OUTPUT_DIR="${REPO_ROOT}/noisy-regression/data/fixed_d2_n16_10m_xy_range3_sigma0p01"
TRAIN_COUNT=10000000
EVAL_COUNT=1024
DIMENSION=2
OBSERVATIONS=16
SIGMA=0.01 # Same standard deviation for independent context and query noise.
CAPACITY=512
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
