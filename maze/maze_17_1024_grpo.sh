#!/usr/bin/env bash
# Maze 17x17 1,024-sample GRPO Single-GPU Training Script

set -euo pipefail

# ============ Configuration ============
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
MODEL_PATH="${REPO_ROOT}/maze/ckpt-1500"
TRAIN_DATA="${REPO_ROOT}/maze/data/maze_17_1024/train.parquet"
VAL_DATA="${REPO_ROOT}/maze/data/maze_17_1024/test.parquet"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"

# Physical GPU assigned to this experiment. Ray sees it as logical GPU 0.
GPU_IDS=(8)
export CUDA_VISIBLE_DEVICES=8

# Mitigate allocator fragmentation from changing rollout sequence lengths.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training hyperparameters
ADVANTAGE_ESTIMATOR=grpo
TRUNCATE_ORDER=64
LR=5e-5
N_ROLLOUTS=128
# The evaluation file contains 128 mazes; N_VAL is responses per maze.
N_VAL=256
# 32 prompts x 128 responses = one 4,096-trajectory batch on the single GPU.
TRAIN_BATCH_SIZE=32
# 1,024 prompts / 32 prompts per step = 32 steps per epoch.
STEPS_PER_EPOCH=32
TOTAL_EPOCHS=200

PROJECT_NAME=maxrl-maze-1024
EXPERIMENT_NAME=${ADVANTAGE_ESTIMATOR}_${N_ROLLOUTS}rollouts

# ============ Ray Setup ============
for required_path in "${VENV_DIR}/bin/activate" "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

source "${VENV_DIR}/bin/activate"

# Load local secrets before Ray starts so its workers inherit them.
# The file is sourced by Bash and must use shell-compatible KEY=value syntax.
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

for gpu_id in "${GPU_IDS[@]}"; do
  if nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU ${gpu_id} already has a compute process; refusing to overlap workloads." >&2
    exit 1
  fi
done

mkdir -p "${CHECKPOINT_DIR}" "${HOME}/tmp"
RAY_TEMP_DIR="$(mktemp -d "${HOME}/tmp/ray-maze-XXXXXX")"

# ray.init() creates and owns this cluster, including cleanup on process exit.
# A local address prevents attaching to another experiment's Ray cluster.
export RAY_ADDRESS=local
export RAY_USAGE_STATS_ENABLED=0
echo "Ray logs and state: ${RAY_TEMP_DIR}"

# ============ Training ============
# Replace the shell so SIGINT/SIGTERM reach Ray's owning Python process.
exec python3 -m verl.trainer.main_ppo \
  "ray_init.ray_dir=${RAY_TEMP_DIR}" \
  algorithm.adv_estimator=${ADVANTAGE_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.pass_k=4 \
  algorithm.truncate_order=${TRUNCATE_ORDER} \
  "data.train_files=${TRAIN_DATA}" \
  "data.val_files=${VAL_DATA}" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=320 \
  data.max_response_length=180 \
  data.apply_chat_template=False \
  "actor_rollout_ref.model.path=${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=${LR} \
  "+actor_rollout_ref.actor.optim.betas=[0.9,0.95]" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.dtype=float16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4096 \
  actor_rollout_ref.rollout.name=hf \
  "actor_rollout_ref.rollout.completion_token_ids=[7]" \
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
  reward_model.reward_manager=prime \
  +reward_model.reward_kwargs.num_processes=64 \
  +reward_model.reward_kwargs.chunksize=64 \
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
