#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

RUN_DIR="${PROJECT_DIR}/high-level/${TEACHER_EXPERIMENT_DIR}/${TEACHER_RUN_NAME}"
LOG_FILE="${REMOTE_ROOT}/remote-logs/${TEACHER_RUN_NAME}.teacher.log"

date '+%Y-%m-%d %H:%M:%S %Z %z'
echo "Run dir: ${RUN_DIR}"
echo "Log: ${LOG_FILE}"
echo
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
echo
if [[ -d "${RUN_DIR}/checkpoints" ]]; then
  find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.pt' -printf '%f %s bytes %TY-%Tm-%Td %TH:%TM:%TS\n' | sort -V | tail -n 10
else
  echo "No checkpoint directory yet: ${RUN_DIR}/checkpoints"
fi
echo
if [[ -f "$LOG_FILE" ]]; then
  tail -n 80 "$LOG_FILE"
else
  echo "No log file yet."
fi
