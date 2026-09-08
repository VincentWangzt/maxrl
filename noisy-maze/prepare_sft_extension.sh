#!/usr/bin/env bash
# Expand the existing SFT corpus while preserving all held-out and RL mazes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
export CUDA_VISIBLE_DEVICES=""
exec "${VENV_DIR}/bin/python" -m noisy_maze.extend_sft \
  --source-dir "${EXPERIMENT_ROOT}/data/${DATASET_TITLE}_sft_100000" \
  --exclude-rl-dir "${EXPERIMENT_ROOT}/data/${DATASET_TITLE}_rl_2048" \
  --train-count "${SFT_TRAIN_COUNT}" \
  --generator-seed 17202610 \
  --noise-seed 71202610
