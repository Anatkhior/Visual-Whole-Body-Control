#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

OUT_DIR="${REMOTE_ROOT}/vbc-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
  fi
}

copy_if_exists "${PROJECT_DIR}/high-level/${TEACHER_EXPERIMENT_DIR}/${TEACHER_RUN_NAME}" "${OUT_DIR}/teacher/${TEACHER_RUN_NAME}"
copy_if_exists "${PROJECT_DIR}/high-level/${STUDENT_EXPERIMENT_DIR}/${STUDENT_RUN_NAME}" "${OUT_DIR}/student/${STUDENT_RUN_NAME}"
copy_if_exists "${REMOTE_ROOT}/remote-logs" "${OUT_DIR}/remote-logs"
copy_if_exists "${REMOTE_ROOT}/agent-log" "${OUT_DIR}/agent-log"

tar -C "$REMOTE_ROOT" -czf "${OUT_DIR}.tar.gz" "$(basename "$OUT_DIR")"
echo "Packed results: ${OUT_DIR}.tar.gz"
