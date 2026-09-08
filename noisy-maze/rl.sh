#!/usr/bin/env bash
# Shared single-GPU launcher for the independent 1,024-sample noisy-maze RL comparison.

set -euo pipefail

if [[ "$#" -ne 2 && "$#" -ne 4 ]]; then
  echo "Usage: $0 {maxrl|grpo|rloo} GPU_ID [--sft-checkpoint-step STEP]" >&2
  exit 2
fi

ADVANTAGE_ESTIMATOR="$1"
GPU_ID="$2"
case "${ADVANTAGE_ESTIMATOR}" in
  maxrl|grpo|rloo) ;;
  *)
    echo "Unsupported advantage estimator: ${ADVANTAGE_ESTIMATOR}" >&2
    exit 2
    ;;
esac

# Choose a checkpoint within the SFT run independently of its training budget.
SFT_CHECKPOINT_STEP=6000
if [[ "$#" -eq 4 ]]; then
  if [[ "$3" != "--sft-checkpoint-step" ]]; then
    echo "Unknown option: $3 (expected --sft-checkpoint-step STEP)" >&2
    exit 2
  fi
  SFT_CHECKPOINT_STEP="$4"
fi
if [[ ! "${SFT_CHECKPOINT_STEP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--sft-checkpoint-step must be a positive integer without leading zeros." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
MODEL_PATH="${SFT_OUTPUT_DIR}/ckpt-${SFT_CHECKPOINT_STEP}"
TRAIN_DATA="${RL_DATA_DIR}/train.parquet"
VAL_DATA="${RL_DATA_DIR}/test.parquet"
CHECKPOINT_DIR="${EXPERIMENT_ROOT}/checkpoints/rl"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRUNCATE_ORDER=64
LR="${LR:-5e-5}"
N_ROLLOUTS=128
# The evaluation file contains 128 mazes; N_VAL is responses per maze.
N_VAL=256
# 32 prompts x 128 responses = one 4,096-trajectory optimizer batch.
TRAIN_BATCH_SIZE=32
# 1,024 prompts / 32 prompts per step = 32 steps per epoch.
STEPS_PER_EPOCH=$((RL_TRAIN_COUNT / TRAIN_BATCH_SIZE))
TOTAL_EPOCHS=200

EXPERIMENT_NAME="${RL_TITLE}-${ADVANTAGE_ESTIMATOR}_${N_ROLLOUTS}rollouts-lr_${LR}-sft_${SFT_TRAIN_COUNT}_${SFT_MAX_STEPS}steps-ckpt_${SFT_CHECKPOINT_STEP}"

for required_path in "${VENV_DIR}/bin/activate" "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

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
if ! nvidia-smi --id="${GPU_ID}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
  echo "GPU ${GPU_ID} does not exist." >&2
  exit 1
fi
if nvidia-smi --id="${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "GPU ${GPU_ID} already has a compute process; refusing to overlap workloads." >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${HOME}/tmp"
RAY_TEMP_DIR="$(mktemp -d "${HOME}/tmp/ray-noisy-maze-XXXXXX")"

export RAY_ADDRESS=local
export RAY_USAGE_STATS_ENABLED=0
echo "Ray logs and state: ${RAY_TEMP_DIR}"

exec python3 -m verl.trainer.main_ppo \
  "ray_init.ray_dir=${RAY_TEMP_DIR}" \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.pass_k=4 \
  algorithm.truncate_order=${TRUNCATE_ORDER} \
  "data.train_files=${TRAIN_DATA}" \
  "data.val_files=${VAL_DATA}" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.apply_chat_template=False \
  "actor_rollout_ref.model.path=${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=${LR} \
  "+actor_rollout_ref.actor.optim.betas=[0.9,0.95]" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.dtype=float16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.micro_batch_size=4096 \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4096 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4096 \
  actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
  actor_rollout_ref.rollout.val_kwargs.n=${N_VAL} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  reward_model.reward_manager=batch \
  "custom_reward_function.path=${EXPERIMENT_ROOT}/noisy_maze/reward.py" \
  custom_reward_function.name=compute_scores \
  algorithm.kl_ctrl.kl_coef=0.0 \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=64 \
  trainer.test_freq=64 \
  trainer.max_actor_ckpt_to_keep=300 \
  "trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.total_training_steps=$((TOTAL_EPOCHS * STEPS_PER_EPOCH))
