#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

CONDA_ENV_NAME="${LOW_CONDA_ENV_NAME:-vbc-low-cu113}"
LOW_GPU="${LOW_GPU:-0}"
LOW_SMOKE_ITERATIONS="${LOW_SMOKE_ITERATIONS:-2}"
LOW_SMOKE_NAME="low-smoke-$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${REMOTE_ROOT}/remote-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${LOW_SMOKE_NAME}.log"

activate_env
cd "${PROJECT_DIR}/low-level/legged_gym/scripts"

echo "Running low-level smoke test. Log: ${LOG_FILE}"
echo "REMOTE_LOW_SMOKE iterations=${LOW_SMOKE_ITERATIONS}" | tee "$LOG_FILE"
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="${LOW_GPU}" timeout 1800s python train.py \
  --headless \
  --debug \
  --exptid "${LOW_SMOKE_NAME}" \
  --proj_name b1z1-low \
  --task b1z1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --observe_gait_commands \
  --max_iterations "${LOW_SMOKE_ITERATIONS}" \
  2>&1 | tee -a "$LOG_FILE"

echo "Low-level smoke test finished."
