#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

echo "REMOTE_ROOT=${REMOTE_ROOT}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo
date '+%Y-%m-%d %H:%M:%S %Z %z'
hostname
uname -a
echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader
else
  echo "nvidia-smi not found" >&2
fi
echo
free -h || true
df -h "$REMOTE_ROOT" || true
echo
command -v conda || true
command -v tmux || true
command -v rsync || true
command -v gcc || true
command -v g++ || true
