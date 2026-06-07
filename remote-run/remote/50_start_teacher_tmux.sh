#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SESSION="${TEACHER_TMUX_SESSION:-vbc_teacher_g${TEACHER_GPU}}"
LOG_DIR="${REMOTE_ROOT}/remote-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TEACHER_RUN_NAME}.teacher.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Use: tmux attach -t ${SESSION}"
  exit 6
fi

CMD=$(cat <<EOF
set -euo pipefail
source '${SCRIPT_DIR}/common.sh'
activate_env
cd '${PROJECT_DIR}/high-level'
echo "Teacher training started at \$(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a '${LOG_FILE}'
echo "Run: ${TEACHER_RUN_NAME}" | tee -a '${LOG_FILE}'
set +e
CUDA_VISIBLE_DEVICES='${TEACHER_GPU}' python train_multistate.py $(teacher_args) 2>&1 | tee -a '${LOG_FILE}'
teacher_exit=\${PIPESTATUS[0]}
set -e
echo "Teacher training python exit code: \${teacher_exit}" | tee -a '${LOG_FILE}'
echo "Teacher training exited at \$(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a '${LOG_FILE}'
exit "\${teacher_exit}"
EOF
)

tmux new-session -d -s "$SESSION" "$CMD"

echo "Started teacher training in tmux session: ${SESSION}"
echo "Log: ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Monitor: bash remote-run/remote/60_monitor_teacher.sh"
