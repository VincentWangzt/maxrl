#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
export CUDA_VISIBLE_DEVICES=""
exec "${VENV_DIR}/bin/python" -m noisy_maze.prepare \
  --size "${MAZE_SIZE}" \
  --noise-fraction "${NOISE_FRACTION}" \
  --rl-train-count "${RL_TRAIN_COUNT}"
