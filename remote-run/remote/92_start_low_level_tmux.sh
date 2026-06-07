#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

CONDA_ENV_NAME="${LOW_CONDA_ENV_NAME:-vbc-low-cu113}"
LOW_GPU="${LOW_GPU:-0}"
LOW_RUN_NAME="${LOW_RUN_NAME:-publiccheckrollrew-retrain-$(date +%Y%m%d-%H%M%S)}"
LOW_NUM_ENVS="${LOW_NUM_ENVS:-6144}"
LOW_MAX_ITERATIONS="${LOW_MAX_ITERATIONS:-45000}"
LOW_TMUX_SESSION="${LOW_TMUX_SESSION:-vbc_low_g${LOW_GPU}}"
LOW_WANDB_MODE="${LOW_WANDB_MODE:-disabled}"
LOW_RESUME_RUN_NAME="${LOW_RESUME_RUN_NAME:-}"
LOW_RESUME_CHECKPOINT="${LOW_RESUME_CHECKPOINT:-}"

LOG_DIR="${REMOTE_ROOT}/remote-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${LOW_RUN_NAME}.low.log"
RUN_DIR="${PROJECT_DIR}/low-level/logs/b1z1-low/${LOW_RUN_NAME}"

resume_args=()
if [[ -n "$LOW_RESUME_RUN_NAME" ]]; then
  resume_args+=(--resume --resumeid "$LOW_RESUME_RUN_NAME")
fi
if [[ -n "$LOW_RESUME_CHECKPOINT" ]]; then
  resume_args+=(--checkpoint "$LOW_RESUME_CHECKPOINT")
fi
RESUME_ARGS=""
if [[ "${#resume_args[@]}" -gt 0 ]]; then
  RESUME_ARGS="$(printf '%q ' "${resume_args[@]}")"
fi

if tmux has-session -t "$LOW_TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists: ${LOW_TMUX_SESSION}" >&2
  echo "Use: tmux attach -t ${LOW_TMUX_SESSION}"
  exit 6
fi

CMD=$(cat <<EOF
set -euo pipefail
source '${SCRIPT_DIR}/common.sh'
CONDA_ENV_NAME='${CONDA_ENV_NAME}'
activate_env
cd '${PROJECT_DIR}/low-level/legged_gym/scripts'
echo "Low-level training started at \$(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a '${LOG_FILE}'
echo "Run: ${LOW_RUN_NAME}" | tee -a '${LOG_FILE}'
echo "Run dir: ${RUN_DIR}" | tee -a '${LOG_FILE}'
echo "num_envs=${LOW_NUM_ENVS} max_iterations=${LOW_MAX_ITERATIONS} gpu=${LOW_GPU} wandb_mode=${LOW_WANDB_MODE} resume_run=${LOW_RESUME_RUN_NAME} resume_checkpoint=${LOW_RESUME_CHECKPOINT}" | tee -a '${LOG_FILE}'
WANDB_MODE='${LOW_WANDB_MODE}' WANDB_DISABLED=true WANDB_SILENT=true CUDA_VISIBLE_DEVICES='${LOW_GPU}' python train.py \
  --headless \
  --exptid '${LOW_RUN_NAME}' \
  --proj_name b1z1-low \
  --task b1z1 \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --observe_gait_commands \
  --num_envs '${LOW_NUM_ENVS}' \
  --max_iterations '${LOW_MAX_ITERATIONS}' \
  ${RESUME_ARGS} \
  2>&1 | tee -a '${LOG_FILE}'
echo "Low-level training exited at \$(date '+%Y-%m-%d %H:%M:%S %Z %z')" | tee -a '${LOG_FILE}'
EOF
)

tmux new-session -d -s "$LOW_TMUX_SESSION" "$CMD"

echo "Started low-level training in tmux session: ${LOW_TMUX_SESSION}"
echo "Run: ${LOW_RUN_NAME}"
echo "Run dir: ${RUN_DIR}"
echo "Log: ${LOG_FILE}"
echo "Attach: tmux attach -t ${LOW_TMUX_SESSION}"
echo "Monitor: LOW_RUN_NAME='${LOW_RUN_NAME}' bash remote-run/remote/93_monitor_low_level.sh"
