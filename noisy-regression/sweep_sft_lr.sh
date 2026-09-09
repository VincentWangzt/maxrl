#!/usr/bin/env bash
set -euo pipefail

# One fresh run per rate, sequentially on the user-selected GPU.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=3
MAX_STEPS=10000
WARMUP_STEPS=500
LEARNING_RATES=(1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4 2e-4 5e-4)
ALLOW_GPU_SHARING=false
RESUME=false
if [[ $# -lt 1 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Usage: bash noisy-regression/sweep_sft_lr.sh SWEEP_NAME [--resume] [--allow-gpu-sharing]" >&2
  exit 2
fi
SWEEP_NAME="$1"
shift
while (( $# )); do
  case "$1" in
    --resume) RESUME=true ;;
    --allow-gpu-sharing) ALLOW_GPU_SHARING=true ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
SWEEP_DIR="${REPO_ROOT}/noisy-regression/checkpoints/${SWEEP_NAME}"
mkdir -p "${REPO_ROOT}/noisy-regression/checkpoints"
exec 9>"${REPO_ROOT}/noisy-regression/checkpoints/.sft_gpu${GPU_ID}.lock"
flock -n 9 || { echo "Another SFT sweep has reserved GPU ${GPU_ID}." >&2; exit 1; }
sweep_settings="$(
  printf 'sweep=%s\ngpu=%s\nmax_steps=%s\nwarmup_steps=%s\n' \
    "${SWEEP_NAME}" "${GPU_ID}" "${MAX_STEPS}" "${WARMUP_STEPS}"
  printf 'learning_rates=%s\n' "${LEARNING_RATES[*]}"
)"
if [[ "${RESUME}" == true ]]; then
  [[ -f "${SWEEP_DIR}/status.tsv" && -d "${SWEEP_DIR}/logs" ]] || {
    echo "No existing sweep to resume: ${SWEEP_DIR}" >&2; exit 1;
  }
  recorded_settings="$(head -n 5 "${SWEEP_DIR}/config.txt")"
  [[ "${recorded_settings}" == "${sweep_settings}" ]] || {
    echo "Existing sweep configuration differs from this launcher." >&2; exit 1;
  }
else
  mkdir "${SWEEP_DIR}"
  mkdir "${SWEEP_DIR}/logs"
  {
    printf '%s\n' "${sweep_settings}"
    git -C "${REPO_ROOT}" rev-parse HEAD
  } >"${SWEEP_DIR}/config.txt"
  printf 'learning_rate\tstatus\tutc\toutput_dir\n' >"${SWEEP_DIR}/status.tsv"
fi
export WANDB_RUN_GROUP="${SWEEP_NAME}"
launch_args=()
if [[ "${ALLOW_GPU_SHARING}" == true ]]; then
  launch_args+=(--allow-gpu-sharing)
fi
printf 'utc=%s resume=%s allow_gpu_sharing=%s git_commit=%s\n' \
  "$(date -u +%FT%TZ)" "${RESUME}" "${ALLOW_GPU_SHARING}" "$(git -C "${REPO_ROOT}" rev-parse HEAD)" \
  >>"${SWEEP_DIR}/launches.log"

for learning_rate in "${LEARNING_RATES[@]}"; do
  run_name="${SWEEP_NAME}_lr${learning_rate}"
  output_dir="${SWEEP_DIR}/lr${learning_rate}"
  if [[ "${RESUME}" == true ]] && awk -F '\t' -v rate="${learning_rate}" \
    '$1 == rate && $2 == "completed" { found = 1 } END { exit !found }' "${SWEEP_DIR}/status.tsv"; then
    [[ -f "${output_dir}/summary.json" && -f "${output_dir}/report.md" && \
       -d "${output_dir}/checkpoint-10000" ]] || {
      echo "Completed run ${run_name} is missing its final artifacts." >&2; exit 1;
    }
    echo "Skipping completed run ${run_name}"
    continue
  fi
  [[ ! -e "${output_dir}" ]] || {
    echo "Unfinished output already exists: ${output_dir}; explicit checkpoint recovery is required." >&2; exit 1;
  }
  printf '%s\trunning\t%s\t%s\n' "${learning_rate}" "$(date -u +%FT%TZ)" "${output_dir}" \
    >>"${SWEEP_DIR}/status.tsv"
  echo "Starting ${run_name} on GPU ${GPU_ID}; log: ${SWEEP_DIR}/logs/lr${learning_rate}.log"
  if bash "${REPO_ROOT}/noisy-regression/sft.sh" "${launch_args[@]}" \
    --gpu-id "${GPU_ID}" --max-steps "${MAX_STEPS}" --warmup-steps "${WARMUP_STEPS}" \
    --learning-rate "${learning_rate}" --min-learning-rate 0 \
    --learning-rate-schedule linear_warmup_constant \
    --run-name "${run_name}" --output-dir "${output_dir}" \
    >>"${SWEEP_DIR}/logs/lr${learning_rate}.log" 2>&1; then
    printf '%s\tcompleted\t%s\t%s\n' "${learning_rate}" "$(date -u +%FT%TZ)" "${output_dir}" \
      >>"${SWEEP_DIR}/status.tsv"
    echo "Completed ${run_name}"
  else
    exit_code=$?
    printf '%s\tfailed:%s\t%s\t%s\n' "${learning_rate}" "${exit_code}" "$(date -u +%FT%TZ)" "${output_dir}" \
      >>"${SWEEP_DIR}/status.tsv"
    echo "Run ${run_name} failed with exit ${exit_code}; stopping sweep." >&2
    exit "${exit_code}"
  fi
done
echo "Completed all ${#LEARNING_RATES[@]} learning rates. Artifacts: ${SWEEP_DIR}"
