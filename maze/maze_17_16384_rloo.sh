#!/usr/bin/env bash
# Maze 17x17 16,384-sample RLOO Training Script

set -euo pipefail

# ============ Configuration ============
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
MODEL_PATH="${REPO_ROOT}/maze/ckpt-1500"
TRAIN_DATA="${REPO_ROOT}/maze/data/maze_17_16384/train.parquet"
VAL_DATA="${REPO_ROOT}/maze/data/maze_17_16384/test.parquet"
CHECKPOINT_DIR="${REPO_ROOT}/checkpoints"

# Physical GPUs assigned to this experiment. Ray sees these as logical GPUs 0-3.
GPU_IDS=(0 1)
export CUDA_VISIBLE_DEVICES=0,1

# Mitigate allocator fragmentation from changing rollout sequence lengths.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use a dedicated Ray address and state directory on the shared server.
RAY_PORT=8341
RAY_DASHBOARD_PORT=8342
RAY_TEMP_DIR="/home/zitongw2/tmp/ray-maxrl-${RAY_PORT}"

# Training hyperparameters
ADVANTAGE_ESTIMATOR=rloo
TRUNCATE_ORDER=64
LR=1e-4
N_ROLLOUTS=128
N_VAL=256
TRAIN_BATCH_SIZE=128

PROJECT_NAME=maxrl-maze-16384
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

for ray_port in "${RAY_PORT}" "${RAY_DASHBOARD_PORT}"; do
  if ss -ltnH "sport = :${ray_port}" | grep -q .; then
    echo "Ray port ${ray_port} is already in use." >&2
    exit 1
  fi
done

if pgrep -u "$(id -u)" -f '[r]aylet|[g]cs_server' >/dev/null; then
  echo "This account already owns a Ray cluster. Stop it or use another account before starting this experiment." >&2
  exit 1
fi

for gpu_id in "${GPU_IDS[@]}"; do
  if nvidia-smi --id="${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
    echo "GPU ${gpu_id} already has a compute process; refusing to overlap workloads." >&2
    exit 1
  fi
done

mkdir -p "${CHECKPOINT_DIR}" "${RAY_TEMP_DIR}"
unset RAY_ADDRESS

ray_started=1
cleanup_ray() {
  if [[ "${ray_started}" -eq 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
    if pgrep -u "$(id -u)" -f '[r]aylet|[g]cs_server' >/dev/null; then
      echo "Ray cleanup left processes running for this account." >&2
    fi
  fi
}
trap cleanup_ray EXIT

ray start --head \
  --port="${RAY_PORT}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --num-gpus=2 \
  --temp-dir="${RAY_TEMP_DIR}" \
  --include-dashboard=false \
  --disable-usage-stats

export RAY_ADDRESS="$(<"${RAY_TEMP_DIR}/ray_current_cluster")"

ray_ready=0
for _ in {1..30}; do
  if ray status 2>&1 | grep -q 'Active:'; then
    ray_ready=1
    break
  fi
  sleep 1
done

if [[ "${ray_ready}" -ne 1 ]]; then
  echo "Ray did not become ready on ${RAY_ADDRESS} within 30 seconds." >&2
  exit 1
fi

ray status

# ============ Training ============
python3 -m verl.trainer.main_ppo \
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
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=128 \
  trainer.test_freq=64 \
  trainer.max_actor_ckpt_to_keep=300 \
  "trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
  trainer.total_epochs=10 \
  trainer.total_training_steps=1280
