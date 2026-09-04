#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/maze_17_16384/$(date +%Y%m%d-%H%M%S)"

mkdir -p "${LOG_DIR}"

scripts=(
  "${REPO_ROOT}/maze/maze_17_16384_maxrl.sh"
  "${REPO_ROOT}/maze/maze_17_16384_grpo.sh"
  "${REPO_ROOT}/maze/maze_17_16384_rloo.sh"
)

overall_status=0
for script in "${scripts[@]}"; do
  script_name="$(basename -- "${script}" .sh)"
  log_file="${LOG_DIR}/${script_name}.log"

  printf '\n[%s] Starting %s\n' "$(date --iso-8601=seconds)" "${script_name}" | tee -a "${LOG_DIR}/queue.log"
  "${script}" 2>&1 | tee "${log_file}"
  script_status=${PIPESTATUS[0]}
  printf '[%s] Finished %s with exit status %s\n' \
    "$(date --iso-8601=seconds)" "${script_name}" "${script_status}" | tee -a "${LOG_DIR}/queue.log"

  if [[ "${script_status}" -ne 0 ]]; then
    overall_status=1
  fi
done

printf '\nQueue finished. Logs: %s\n' "${LOG_DIR}" | tee -a "${LOG_DIR}/queue.log"
exit "${overall_status}"
