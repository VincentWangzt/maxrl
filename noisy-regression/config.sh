#!/usr/bin/env bash
# Shared identity for canonical preparation, training and evaluation.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${REPO_ROOT}/.env"
DATA_NAME="fixed_d2_n64_10m_sep_eoo_range4_sigma0p1"
DATA_DIR="${REPO_ROOT}/noisy-regression/data/${DATA_NAME}"
RUN_PREFIX="canonical_d2_n64_10m_sep_eoo_range4_sigma0p1"
CANONICAL_RUN_NAME="${RUN_PREFIX}_bs1024_lr1e-4"
