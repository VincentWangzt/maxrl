#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/noisy-regression/data/fixed_d4_n16_10m_xy_sigma0p1"
RUN_NAME="qwen2_1m_fixed10m_xy_sft_150000_bs64_lr1e-4_sigma0p1"
CHECKPOINT="${REPO_ROOT}/noisy-regression/checkpoints/${RUN_NAME}/checkpoint-150000"
OUTPUT_DIR="${REPO_ROOT}/noisy-regression/checkpoints/${RUN_NAME}/reevaluation-final"
GPU_ID=1
DEVICE="cuda:0"
PRECISION="bf16"
EVAL_BATCH_SIZE=32
CPU_THREADS=4

source "${VENV_DIR}/bin/activate"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
nvidia-smi --id="${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null
if nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "Selected GPU ${GPU_ID} has a compute process; refusing to overlap workloads." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}/noisy-regression:${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "${REPO_ROOT}"
exec python -m noisy_regression.evaluate --data "${DATA_DIR}" --checkpoint "${CHECKPOINT}" --output "${OUTPUT_DIR}" \
  --device "${DEVICE}" --precision "${PRECISION}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" --cpu-threads "${CPU_THREADS}"
