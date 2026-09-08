#!/usr/bin/env bash
# Train the noisy-maze model for exactly 6,000 optimizer steps.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 GPU_ID" >&2
  exit 2
fi

GPU_ID="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
TRAIN_DATA="${SFT_DATA_DIR}/train.json"
VAL_DATA="${SFT_DATA_DIR}/test.json"
OUTPUT_DIR="${SFT_OUTPUT_DIR}"

for required_path in "${VENV_DIR}/bin/activate" "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing SFT output: ${OUTPUT_DIR}" >&2
  exit 1
fi

if ! nvidia-smi --id="${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "GPU ${GPU_ID} does not exist." >&2
  exit 1
fi
if nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "GPU ${GPU_ID} already has a compute process; refusing to overlap workloads." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Export it or add it to ${ENV_FILE}." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python3 -m noisy_maze.sft \
  --train_data "${TRAIN_DATA}" \
  --val_data "${VAL_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_position_embeddings "${MAX_LENGTH}" \
  --batch_size "${SFT_BATCH_SIZE}" \
  --micro_batch_size "${SFT_BATCH_SIZE}" \
  --learning_rate 5e-4 \
  --lr_scheduler constant \
  --num_epochs 1 \
  --max_steps "${SFT_MAX_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --save_steps 500 \
  --eval_steps 500 \
  --use_generative_eval \
  --eval_samples 128 \
  --n_samples_per_prompt 256 \
  --eval_generation_batch_size 32 \
  --eval_temperature 1.0 \
  --eval_max_new_tokens "${MAX_RESPONSE_LENGTH}" \
  --project_name "${PROJECT_NAME}" \
  --experiment_name "${SFT_TITLE}-constant-lr-5e-4-${SFT_MAX_STEPS}steps" \
  --use_wandb
