#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

LOW_RUN_NAME="${LOW_RUN_NAME:?Set LOW_RUN_NAME to the low-level run name.}"
LOW_CHECKPOINT_STEP="${LOW_CHECKPOINT_STEP:-42000}"
LOW_TMUX_SESSION="${LOW_TMUX_SESSION:-vbc_low_g${LOW_GPU:-0}}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
TEACHER_GPU="${TEACHER_GPU:-1}"
TEACHER_RUN_NAME="${TEACHER_RUN_NAME:-teacher-after-${LOW_RUN_NAME}-${LOW_CHECKPOINT_STEP}}"
TEACHER_TIMESTEPS="${TEACHER_TIMESTEPS:-60000}"
VBC_USE_WANDB="${VBC_USE_WANDB:-0}"

LOW_CKPT="${PROJECT_DIR}/low-level/logs/b1z1-low/${LOW_RUN_NAME}/model_${LOW_CHECKPOINT_STEP}.pt"
LOG_DIR="${REMOTE_ROOT}/remote-logs"
mkdir -p "$LOG_DIR"
WATCH_LOG="${LOG_DIR}/wait-${LOW_RUN_NAME}-then-teacher.log"

echo "Watcher started at $(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a "$WATCH_LOG"
echo "Waiting for: ${LOW_CKPT}" | tee -a "$WATCH_LOG"
echo "Teacher run: ${TEACHER_RUN_NAME} on GPU ${TEACHER_GPU}" | tee -a "$WATCH_LOG"

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S %Z %z')"
  if [[ -f "$LOW_CKPT" ]]; then
    echo "${now} found checkpoint: ${LOW_CKPT}" | tee -a "$WATCH_LOG"
    LOW_RUN_NAME="$LOW_RUN_NAME" LOW_CHECKPOINT_STEP="$LOW_CHECKPOINT_STEP" bash "${SCRIPT_DIR}/94_prepare_teacher_from_low.sh" 2>&1 | tee -a "$WATCH_LOG"
    TEACHER_GPU="$TEACHER_GPU" TEACHER_RUN_NAME="$TEACHER_RUN_NAME" TEACHER_TIMESTEPS="$TEACHER_TIMESTEPS" VBC_USE_WANDB="$VBC_USE_WANDB" bash "${SCRIPT_DIR}/50_start_teacher_tmux.sh" 2>&1 | tee -a "$WATCH_LOG"
    echo "Watcher finished at $(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a "$WATCH_LOG"
    exit 0
  fi

  if ! tmux has-session -t "$LOW_TMUX_SESSION" 2>/dev/null; then
    echo "${now} low-level tmux session is not running and checkpoint is missing: ${LOW_TMUX_SESSION}" | tee -a "$WATCH_LOG"
    exit 7
  fi

  echo "${now} checkpoint not ready; sleeping ${WAIT_INTERVAL_SECONDS}s" | tee -a "$WATCH_LOG"
  sleep "$WAIT_INTERVAL_SECONDS"
done
