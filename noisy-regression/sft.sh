#!/usr/bin/env bash
set -euo pipefail

# Defaults reproduce the current 80K-step n=64 experiment; the sweep passes explicit overrides.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/noisy-regression/data/fixed_d2_n64_10m_xy_range3_sigma0p001"
RUN_NAME=""
OUTPUT_DIR=""
RESUME_CHECKPOINT="" # To resume, set a retained checkpoint AND a new OUTPUT_DIR.
GPU_ID=0
ALLOW_GPU_SHARING=false
DEVICE="cuda:0"
PRECISION="bf16"
BATCH_SIZE=128
MICRO_BATCH_SIZE=128
MAX_STEPS=80000
EVAL_INTERVAL=500
LEARNING_RATE=1e-4
MIN_LEARNING_RATE=1e-5
LEARNING_RATE_SCHEDULE="linear_warmup_cosine_decay"
BETA1=0.9
BETA2=0.95
WEIGHT_DECAY=0.01
OPTIMIZER_EPSILON=1e-8
WARMUP_STEPS=1600
MAX_GRAD_NORM=none
EVAL_BATCH_SIZE=32
CPU_THREADS=4
LOG_INTERVAL=10
USE_WANDB=true
PROJECT_NAME="noisy-regression-sft"
MODEL_CONFIG_JSON='{
  "vocab_size": 20, "hidden_size": 128, "num_hidden_layers": 4,
  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 512,
  "max_position_embeddings": 1024, "hidden_act": "silu", "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0, "tie_word_embeddings": true, "attention_dropout": 0.0,
  "use_sliding_window": false, "sliding_window": null,
  "bos_token_id": 19, "pad_token_id": 18, "eos_token_id": null
}'

while (( $# )); do
  case "$1" in
    --allow-gpu-sharing) ALLOW_GPU_SHARING=true; shift ;;
    --gpu-id|--max-steps|--learning-rate|--min-learning-rate|--learning-rate-schedule|--warmup-steps|--max-grad-norm|--run-name|--output-dir)
      [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --gpu-id) GPU_ID="$2" ;;
        --max-steps) MAX_STEPS="$2" ;;
        --learning-rate) LEARNING_RATE="$2" ;;
        --min-learning-rate) MIN_LEARNING_RATE="$2" ;;
        --learning-rate-schedule) LEARNING_RATE_SCHEDULE="$2" ;;
        --warmup-steps) WARMUP_STEPS="$2" ;;
        --max-grad-norm) MAX_GRAD_NORM="$2" ;;
        --run-name) RUN_NAME="$2" ;;
        --output-dir) OUTPUT_DIR="$2" ;;
      esac
      shift 2
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "${RUN_NAME}" ]]; then
  clip_label="clip${MAX_GRAD_NORM}"
  if [[ "${MAX_GRAD_NORM}" == none ]]; then
    clip_label=noclip
  fi
  RUN_NAME="qwen2_1m_d2_n64_10m_xy_range3_sft_${MAX_STEPS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}_minlr${MIN_LEARNING_RATE}_warmup${WARMUP_STEPS}_${clip_label}_sigma0p001"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${REPO_ROOT}/noisy-regression/checkpoints/${RUN_NAME}"
fi
EXPERIMENT_NAME="${RUN_NAME}"

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
gpu_processes="$(nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ "${gpu_processes}" =~ [0-9] ]]; then
  if [[ "${ALLOW_GPU_SHARING}" == true ]]; then
    echo "Sharing GPU ${GPU_ID} with existing compute processes (--allow-gpu-sharing)."
  else
    echo "Selected GPU ${GPU_ID} has a compute process; refusing to overlap workloads." >&2
    exit 1
  fi
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
  --learning-rate "${LEARNING_RATE}" --min-learning-rate "${MIN_LEARNING_RATE}" \
  --learning-rate-schedule "${LEARNING_RATE_SCHEDULE}" --beta1 "${BETA1}" --beta2 "${BETA2}" \
  --weight-decay "${WEIGHT_DECAY}" --optimizer-epsilon "${OPTIMIZER_EPSILON}" \
  --warmup-steps "${WARMUP_STEPS}" --max-grad-norm "${MAX_GRAD_NORM}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --cpu-threads "${CPU_THREADS}" --log-interval "${LOG_INTERVAL}"
exec python -m noisy_regression.report --run "${OUTPUT_DIR}"
