#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/noisy-regression/data/fixed_d4_n16_10m_xy_sigma0p1"
RUN_NAME="bayesian_continuous_10m_xy_sigma0p1"
OUTPUT_DIR="${REPO_ROOT}/noisy-regression/checkpoints/${RUN_NAME}"
PROJECT_NAME="noisy-regression-sft"
USE_WANDB=true
CPU_THREADS=4

source "${VENV_DIR}/bin/activate"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
if [[ "${USE_WANDB}" == true && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Export it or add it to ${ENV_FILE}." >&2
  exit 1
fi
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "Refusing to overwrite ${OUTPUT_DIR}" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${REPO_ROOT}/noisy-regression:${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
cd "${REPO_ROOT}"
tracking_args=()
if [[ "${USE_WANDB}" == true ]]; then
  tracking_args+=(--use-wandb)
fi
exec python -m noisy_regression.evaluate_baseline --method bayesian \
  --data "${DATA_DIR}" --output "${OUTPUT_DIR}" \
  --project-name "${PROJECT_NAME}" --experiment-name "${RUN_NAME}" "${tracking_args[@]}"
