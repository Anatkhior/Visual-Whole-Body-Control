#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -z "$STUDENT_TEACHER_CKPT" ]]; then
  echo "Set STUDENT_TEACHER_CKPT to a trained teacher checkpoint before starting student." >&2
  exit 4
fi

SESSION="${STUDENT_TMUX_SESSION:-vbc_student_g${STUDENT_GPU}}"
LOG_DIR="${REMOTE_ROOT}/remote-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${STUDENT_RUN_NAME}.student.log"
STUDENT_DISPLAY="${STUDENT_DISPLAY:-${DISPLAY:-}}"

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
echo "Student training started at \$(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a '${LOG_FILE}'
echo "Run: ${STUDENT_RUN_NAME}" | tee -a '${LOG_FILE}'
echo "Display: ${STUDENT_DISPLAY:-<unset>}" | tee -a '${LOG_FILE}'
echo "Checkpoint: ${STUDENT_CHECKPOINT:-<none>}" | tee -a '${LOG_FILE}'
if [[ -n '${STUDENT_DISPLAY}' ]]; then
  export DISPLAY='${STUDENT_DISPLAY}'
fi
set +e
CUDA_VISIBLE_DEVICES='${STUDENT_GPU}' python train_multi_bc_deter.py $(student_args) 2>&1 | tee -a '${LOG_FILE}'
status=\${PIPESTATUS[0]}
set -e
echo "Student training python exit code: \${status}" | tee -a '${LOG_FILE}'
echo "Student training exited at \$(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a '${LOG_FILE}'
exit \${status}
EOF
)

tmux new-session -d -s "$SESSION" "$CMD"

echo "Started student training in tmux session: ${SESSION}"
echo "Log: ${LOG_FILE}"
echo "Attach: tmux attach -t ${SESSION}"
