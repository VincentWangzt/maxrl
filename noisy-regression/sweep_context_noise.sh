#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"

# Explicit physical GPU assignments for the d=2 context/noise cross product.
GPUS=(0 1 2 3)
CONTEXTS=(16 16 32 32)
SIGMAS=(0.001 0.25 0.001 0.25)
TRAIN_COUNT=10000000
EVAL_COUNT=1024
DIMENSION=2
CAPACITY=1024
CPU_THREADS=4
MAX_STEPS=20000
SWEEP_NAME="${1:-context_noise_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ $# -le 1 && "${SWEEP_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "Usage: bash noisy-regression/sweep_context_noise.sh [SWEEP_NAME]" >&2; exit 2;
}
LOG_DIR="${REPO_ROOT}/noisy-regression/logs/${SWEEP_NAME}"
[[ ! -e "${LOG_DIR}" ]] || { echo "Refusing to overwrite ${LOG_DIR}" >&2; exit 1; }
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "Missing server virtualenv." >&2; exit 1; }
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
[[ -n "${WANDB_API_KEY:-}" ]] || { echo "WANDB_API_KEY is not set." >&2; exit 1; }

data_dirs=()
run_names=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  identity="d${DIMENSION}_n${CONTEXTS[$i]}_10m_sep_eoo_range4_sigma${SIGMAS[$i]//./p}"
  data_dirs+=("${REPO_ROOT}/noisy-regression/data/fixed_${identity}")
  run_names+=("canonical_${identity}_bs1024_lr1e-4")
  for output in "${data_dirs[$i]}" "${REPO_ROOT}/noisy-regression/checkpoints/${run_names[$i]}"; do
    [[ ! -e "${output}" ]] || { echo "Refusing to overwrite ${output}" >&2; exit 1; }
  done
  nvidia-smi --id="${gpu}" --query-gpu=index --format=csv,noheader,nounits >/dev/null
  gpu_processes="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ ! "${gpu_processes}" =~ [0-9] ]] || { echo "GPU ${gpu} has a compute process." >&2; exit 1; }
done

mkdir -p "${LOG_DIR}"
git -C "${REPO_ROOT}" rev-parse HEAD >"${LOG_DIR}/git_commit.txt"
printf 'gpu\tobservations\tsigma\ttrain_count\teval_count\tmax_steps\tpid\tdata_dir\trun_name\n' >"${LOG_DIR}/runs.tsv"
export WANDB_RUN_GROUP="${SWEEP_NAME}"
export PYTHONPATH="${REPO_ROOT}/noisy-regression:${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
cd "${REPO_ROOT}"
pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  (
    trap 'code=$?; printf "%s\n" "${code}" >"${LOG_DIR}/gpu${gpu}.exit"' EXIT
    printf 'preparing\n' >"${LOG_DIR}/gpu${gpu}.phase"
    CUDA_VISIBLE_DEVICES="" "${VENV_DIR}/bin/python" -m noisy_regression.data \
      --output "${data_dirs[$i]}" --train-count "${TRAIN_COUNT}" --eval-count "${EVAL_COUNT}" \
      --dimension "${DIMENSION}" --observations "${CONTEXTS[$i]}" \
      --sigma "${SIGMAS[$i]}" --capacity "${CAPACITY}" \
      >"${LOG_DIR}/prepare_gpu${gpu}.log" 2>&1
    printf 'training\n' >"${LOG_DIR}/gpu${gpu}.phase"
    # sft.sh rechecks the selected GPU after CPU preparation and verifies pool hashes.
    bash "${REPO_ROOT}/noisy-regression/sft.sh" --gpu-id "${gpu}" \
      --data-dir "${data_dirs[$i]}" --run-name "${run_names[$i]}" --max-steps "${MAX_STEPS}" \
      >"${LOG_DIR}/gpu${gpu}.log" 2>&1
    printf 'complete\n' >"${LOG_DIR}/gpu${gpu}.phase"
  ) &
  pids+=("$!")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${gpu}" "${CONTEXTS[$i]}" "${SIGMAS[$i]}" "${TRAIN_COUNT}" "${EVAL_COUNT}" \
    "${MAX_STEPS}" "$!" "${data_dirs[$i]}" "${run_names[$i]}" >>"${LOG_DIR}/runs.tsv"
  echo "Started preparation for ${run_names[$i]}; then training on GPU ${gpu}."
done
failed=0
for i in "${!pids[@]}"; do
  code=0
  wait "${pids[$i]}" || code=$?
  echo "GPU ${GPUS[$i]} pipeline exited with code ${code}."
  if (( code != 0 )); then failed=1; fi
done
exit "${failed}"
