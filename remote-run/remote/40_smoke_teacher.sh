#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_env
cd "${PROJECT_DIR}/high-level"

SMOKE_NAME="teacher-smoke-${TEACHER_SMOKE_ENVS}env-${TEACHER_SMOKE_TIMESTEPS}step-$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${REMOTE_ROOT}/remote-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${SMOKE_NAME}.log"
CFG_FILE="data/cfg/b1z1_pickmulti.yaml"
CFG_BACKUP="$(mktemp)"

cp "$CFG_FILE" "$CFG_BACKUP"
restore_cfg() {
  cp "$CFG_BACKUP" "$CFG_FILE"
  rm -f "$CFG_BACKUP"
}
trap restore_cfg EXIT

echo "Running teacher smoke test. Log: ${LOG_FILE}"
python - <<'PY' "$CFG_FILE" "$TEACHER_SMOKE_ENVS"
import sys
import yaml

path = sys.argv[1]
num_envs = int(sys.argv[2])
with open(path, "r") as f:
    cfg = yaml.safe_load(f)
cfg["env"]["numEnvs"] = num_envs
with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "REMOTE_TEACHER_SMOKE num_envs=${TEACHER_SMOKE_ENVS} timesteps=${TEACHER_SMOKE_TIMESTEPS}" | tee "$LOG_FILE"
CUDA_VISIBLE_DEVICES="${TEACHER_GPU}" timeout 1800s python train_multistate.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps "${TEACHER_SMOKE_TIMESTEPS}" \
  --headless \
  --task B1Z1PickMulti \
  --experiment_dir experiments-smoke \
  --wandb_name "${SMOKE_NAME}" \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --stop_pick \
  2>&1 | tee -a "$LOG_FILE"

echo "Teacher smoke test finished."
