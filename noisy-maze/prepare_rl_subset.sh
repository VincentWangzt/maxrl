#!/usr/bin/env bash
# Preserve the existing evaluation set when reducing an already generated RL split.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
export CUDA_VISIBLE_DEVICES=""
exec "${VENV_DIR}/bin/python" -m noisy_maze.curate \
  --source-dir "${EXPERIMENT_ROOT}/data/${DATASET_TITLE}_rl_2048" \
  --train-count "${RL_TRAIN_COUNT}" \
  --seed 1024
