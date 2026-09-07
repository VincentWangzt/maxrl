#!/usr/bin/env bash
# Shared artifact names and context budgets for this independent experiment.
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${EXPERIMENT_ROOT}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
PROJECT_NAME="noisy_maze_maxrl_17x17"
MAZE_SIZE="${MAZE_SIZE:-17}"
NOISE_FRACTION="${NOISE_FRACTION:-0.1}"
if [[ ! "${NOISE_FRACTION}" =~ ^(0|1|0\.[0-9]*[1-9])$ ]]; then
  echo "Use a canonical noise fraction in [0,1], e.g. 0.1 (without trailing zeros)." >&2
  exit 2
fi
case "${MAZE_SIZE}" in
  17) MAX_PROMPT_LENGTH=320; MAX_RESPONSE_LENGTH=180; MAX_LENGTH=512; ACTOR_MICRO_BATCH_SIZE=4096 ;;
  23) MAX_PROMPT_LENGTH=576; MAX_RESPONSE_LENGTH=256; MAX_LENGTH=1024; ACTOR_MICRO_BATCH_SIZE=2048 ;;
  *) echo "Launchers support MAZE_SIZE=17 or 23." >&2; exit 2 ;;
esac
DATASET_TITLE="noisy_maze_${MAZE_SIZE}_noise_${NOISE_FRACTION}"
SFT_TITLE="${DATASET_TITLE}_sft_100000"
SFT_MAX_STEPS=6000
RL_TRAIN_COUNT=1024
RL_TITLE="${DATASET_TITLE}_rl_${RL_TRAIN_COUNT}"
SFT_DATA_DIR="${EXPERIMENT_ROOT}/data/${SFT_TITLE}"
RL_DATA_DIR="${EXPERIMENT_ROOT}/data/${RL_TITLE}"
SFT_OUTPUT_DIR="${EXPERIMENT_ROOT}/checkpoints/${SFT_TITLE}_${SFT_MAX_STEPS}steps"
export PYTHONPATH="${EXPERIMENT_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
