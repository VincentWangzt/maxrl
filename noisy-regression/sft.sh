#!/usr/bin/env bash
set -euo pipefail

# GPU 1 and the 10,000,000-example pool were explicitly selected by the user.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/noisy-regression/data/fixed_d4_n16_10m_xy_sigma0p1"
RUN_NAME="qwen2_1m_fixed10m_xy_sft_150000_bs64_lr1e-4_sigma0p1"
OUTPUT_DIR="${REPO_ROOT}/noisy-regression/checkpoints/${RUN_NAME}"
RESUME_CHECKPOINT="" # To resume, set a retained checkpoint AND a new OUTPUT_DIR.
GPU_ID=1
DEVICE="cuda:0"
PRECISION="bf16"
BATCH_SIZE=64
MICRO_BATCH_SIZE=64
MAX_STEPS=150000
EVAL_INTERVAL=500
LEARNING_RATE=1e-4
BETA1=0.9
BETA2=0.95
WEIGHT_DECAY=0.01
OPTIMIZER_EPSILON=1e-8
WARMUP_STEPS=200
MAX_GRAD_NORM=1.0
TRAIN_EVAL_SIZE=1024
EVAL_BATCH_SIZE=32
CPU_THREADS=4
LOG_INTERVAL=10
USE_WANDB=true
PROJECT_NAME="noisy-regression-sft"
EXPERIMENT_NAME="${RUN_NAME}"
MODEL_CONFIG_JSON='{
  "vocab_size": 20, "hidden_size": 128, "num_hidden_layers": 4,
  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 512,
  "max_position_embeddings": 512, "hidden_act": "silu", "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0, "tie_word_embeddings": true, "attention_dropout": 0.0,
  "use_sliding_window": false, "sliding_window": null,
  "bos_token_id": 19, "pad_token_id": 18, "eos_token_id": null
}'

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
export WANDB_MODE=online
cd "${REPO_ROOT}"
resume_args=()
tracking_args=(--project-name "${PROJECT_NAME}" --experiment-name "${EXPERIMENT_NAME}")
if [[ "${USE_WANDB}" == true ]]; then
  tracking_args+=(--use-wandb)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  resume_args=(--resume "${RESUME_CHECKPOINT}")
fi
python -m noisy_regression.train --data "${DATA_DIR}" --output "${OUTPUT_DIR}" "${resume_args[@]}" "${tracking_args[@]}" \
  --model-config-json "${MODEL_CONFIG_JSON}" \
  --device "${DEVICE}" --precision "${PRECISION}" \
  --batch-size "${BATCH_SIZE}" --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" --eval-interval "${EVAL_INTERVAL}" \
  --learning-rate "${LEARNING_RATE}" --beta1 "${BETA1}" --beta2 "${BETA2}" \
  --weight-decay "${WEIGHT_DECAY}" --optimizer-epsilon "${OPTIMIZER_EPSILON}" \
  --warmup-steps "${WARMUP_STEPS}" --max-grad-norm "${MAX_GRAD_NORM}" \
  --train-eval-size "${TRAIN_EVAL_SIZE}" --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --cpu-threads "${CPU_THREADS}" --log-interval "${LOG_INTERVAL}"
exec python -m noisy_regression.report --run "${OUTPUT_DIR}"
