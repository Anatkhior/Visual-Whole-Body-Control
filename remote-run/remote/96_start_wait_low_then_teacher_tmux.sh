#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

LOW_RUN_NAME="${LOW_RUN_NAME:?Set LOW_RUN_NAME to the low-level run name.}"
LOW_CHECKPOINT_STEP="${LOW_CHECKPOINT_STEP:-42000}"
WATCH_TMUX_SESSION="${WATCH_TMUX_SESSION:-vbc_wait_low_teacher}"
TEACHER_GPU="${TEACHER_GPU:-1}"
TEACHER_RUN_NAME="${TEACHER_RUN_NAME:-teacher-after-${LOW_RUN_NAME}-${LOW_CHECKPOINT_STEP}}"

if tmux has-session -t "$WATCH_TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists: ${WATCH_TMUX_SESSION}" >&2
  echo "Use: tmux attach -t ${WATCH_TMUX_SESSION}"
  exit 6
fi

CMD=$(cat <<EOF
set -euo pipefail
cd '${REMOTE_ROOT}'
LOW_RUN_NAME='${LOW_RUN_NAME}' \
LOW_CHECKPOINT_STEP='${LOW_CHECKPOINT_STEP}' \
TEACHER_GPU='${TEACHER_GPU}' \
TEACHER_RUN_NAME='${TEACHER_RUN_NAME}' \
bash '${SCRIPT_DIR}/95_wait_low_then_start_teacher.sh'
EOF
)

tmux new-session -d -s "$WATCH_TMUX_SESSION" "$CMD"

echo "Started low-to-teacher watcher in tmux session: ${WATCH_TMUX_SESSION}"
echo "Low run: ${LOW_RUN_NAME}"
echo "Checkpoint step: ${LOW_CHECKPOINT_STEP}"
echo "Teacher run: ${TEACHER_RUN_NAME}"
echo "Log: ${REMOTE_ROOT}/remote-logs/wait-${LOW_RUN_NAME}-then-teacher.log"
