#!/usr/bin/env bash
set -euo pipefail

# GPU 1 and the 100,000-example pool were explicitly selected by the user.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
DATA_DIR="${REPO_ROOT}/noisy-regression/data/fixed_d4_n16_100k"
RUN_NAME="qwen2_1m_fixed100k_sft_10000"
OUTPUT_DIR="${REPO_ROOT}/noisy-regression/checkpoints/${RUN_NAME}"
RESUME_CHECKPOINT="" # To resume, set a retained checkpoint AND a new OUTPUT_DIR.
GPU_ID=1
DEVICE="cuda:0"
PRECISION="bf16"
BATCH_SIZE=64
MICRO_BATCH_SIZE=16
MAX_STEPS=10000
EVAL_INTERVAL=500
LEARNING_RATE=5e-4
BETA1=0.9
BETA2=0.95
WEIGHT_DECAY=0.01
OPTIMIZER_EPSILON=1e-8
WARMUP_STEPS=200
MAX_GRAD_NORM=1.0
SEED=3141
ORDER_SEED=1618
SUBSET_SEED=5772
SAMPLING_SEED=8119
TRAIN_EVAL_SIZE=1024
GENERATION_SUBSET_SIZE=128
EVAL_BATCH_SIZE=32
GENERATION_BATCH_SIZE=32
SAMPLES=256
CPU_THREADS=4
LOG_INTERVAL=10
MODEL_CONFIG_JSON='{
  "vocab_size": 22, "hidden_size": 128, "num_hidden_layers": 4,
  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 512,
  "max_position_embeddings": 512, "hidden_act": "silu", "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0, "tie_word_embeddings": true, "attention_dropout": 0.0,
  "use_sliding_window": false, "sliding_window": null,
  "bos_token_id": 21, "pad_token_id": 20, "eos_token_id": null
}'

source "${VENV_DIR}/bin/activate"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
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
cd "${REPO_ROOT}"
resume_args=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  resume_args=(--resume "${RESUME_CHECKPOINT}")
fi
python -m noisy_regression.train --data "${DATA_DIR}" --output "${OUTPUT_DIR}" "${resume_args[@]}" \
  --model-config-json "${MODEL_CONFIG_JSON}" \
  --device "${DEVICE}" --precision "${PRECISION}" \
  --batch-size "${BATCH_SIZE}" --micro-batch-size "${MICRO_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" --eval-interval "${EVAL_INTERVAL}" \
  --learning-rate "${LEARNING_RATE}" --beta1 "${BETA1}" --beta2 "${BETA2}" \
  --weight-decay "${WEIGHT_DECAY}" --optimizer-epsilon "${OPTIMIZER_EPSILON}" \
  --warmup-steps "${WARMUP_STEPS}" --max-grad-norm "${MAX_GRAD_NORM}" \
  --seed "${SEED}" --order-seed "${ORDER_SEED}" --subset-seed "${SUBSET_SEED}" --sampling-seed "${SAMPLING_SEED}" \
  --train-eval-size "${TRAIN_EVAL_SIZE}" --generation-subset-size "${GENERATION_SUBSET_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --samples "${SAMPLES}" --cpu-threads "${CPU_THREADS}" --log-interval "${LOG_INTERVAL}"
exec python -m noisy_regression.report --run "${OUTPUT_DIR}"
