#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

LOW_RUN_NAME="${LOW_RUN_NAME:?Set LOW_RUN_NAME to the low-level run name.}"
LOW_TMUX_SESSION="${LOW_TMUX_SESSION:-vbc_low_g${LOW_GPU:-0}}"
RUN_DIR="${PROJECT_DIR}/low-level/logs/b1z1-low/${LOW_RUN_NAME}"
LOG_FILE="${REMOTE_ROOT}/remote-logs/${LOW_RUN_NAME}.low.log"

date '+%Y-%m-%d %H:%M:%S %Z %z'
echo "Session: ${LOW_TMUX_SESSION}"
tmux has-session -t "$LOW_TMUX_SESSION" 2>/dev/null && echo "tmux: running" || echo "tmux: not running"
echo "Run dir: ${RUN_DIR}"
echo "Log: ${LOG_FILE}"
echo
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
echo
if [[ -d "$RUN_DIR" ]]; then
  find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f %s bytes %TY-%Tm-%Td %TH:%TM:%TS\n' | sort -V | tail -n 12
else
  echo "No run dir yet: ${RUN_DIR}"
fi
echo
if [[ -f "$LOG_FILE" ]]; then
  grep -E "Learning iteration|Mean reward|Total timesteps|Low-level training started|Low-level training exited" "$LOG_FILE" | tail -n 30 || true
  echo
  tail -n 80 "$LOG_FILE"
else
  echo "No log file yet."
fi
