#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

# One-factor changes; physical GPUs explicitly assigned for this experiment.
GPUS=(2 6 7 8 9)
BATCHES=(1024 512 2048 1024 1024)
RATES=(1e-4 1e-4 1e-4 5e-5 2e-4)
SWEEP_NAME="${1:-canonical_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ $# -le 1 && "${SWEEP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "Usage: bash noisy-regression/sweep_sft.sh [SWEEP_NAME]" >&2; exit 2;
}
LOG_DIR="${REPO_ROOT}/noisy-regression/logs/${SWEEP_NAME}"
[[ ! -e "${LOG_DIR}" ]] || { echo "Refusing to overwrite ${LOG_DIR}" >&2; exit 1; }
[[ -f "${DATA_DIR}/metadata.json" ]] || { echo "Run noisy-regression/prepare.sh first." >&2; exit 1; }
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is not set." >&2; exit 1; }
# Check all requested devices and output paths before launching any run.
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  run_name="${RUN_PREFIX}_bs${BATCHES[$i]}_lr${RATES[$i]}"
  [[ ! -e "${REPO_ROOT}/noisy-regression/checkpoints/${run_name}" ]] || {
    echo "Output exists for ${run_name}." >&2; exit 1;
  }
  nvidia-smi --id="${gpu}" --query-gpu=index --format=csv,noheader,nounits >/dev/null
  gpu_processes="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ ! "${gpu_processes}" =~ [0-9] ]] || { echo "GPU ${gpu} has a compute process." >&2; exit 1; }
done
mkdir -p "${LOG_DIR}"
git -C "${REPO_ROOT}" rev-parse HEAD >"${LOG_DIR}/git_commit.txt"
printf 'gpu\tbatch_size\tlearning_rate\tpid\trun_name\n' >"${LOG_DIR}/runs.tsv"
printf 'gpu\texit_code\n' >"${LOG_DIR}/exit_status.tsv"
export WANDB_RUN_GROUP="${SWEEP_NAME}"
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  run_name="${RUN_PREFIX}_bs${BATCHES[$i]}_lr${RATES[$i]}"
  bash "${REPO_ROOT}/noisy-regression/sft.sh" --gpu-id "${gpu}" \
    --batch-size "${BATCHES[$i]}" --learning-rate "${RATES[$i]}" \
    >"${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  printf '%s\t%s\t%s\t%s\t%s\n' "${gpu}" "${BATCHES[$i]}" "${RATES[$i]}" "$!" "${run_name}" \
    >>"${LOG_DIR}/runs.tsv"
  echo "Started ${run_name} on GPU ${gpu}; log: ${LOG_DIR}/gpu${gpu}.log"
done
failed=0
for i in "${!pids[@]}"; do
  code=0
  wait "${pids[$i]}" || code=$?
  printf '%s\t%s\n' "${GPUS[$i]}" "${code}" >>"${LOG_DIR}/exit_status.tsv"
  if (( code != 0 )); then failed=1; fi
done
exit "${failed}"
