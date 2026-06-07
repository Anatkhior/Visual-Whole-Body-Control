#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

LOW_RUN_NAME="${LOW_RUN_NAME:?Set LOW_RUN_NAME to the low-level run name.}"
LOW_CHECKPOINT_STEP="${LOW_CHECKPOINT_STEP:-42000}"
LOW_CKPT="${LOW_CKPT:-${PROJECT_DIR}/low-level/logs/b1z1-low/${LOW_RUN_NAME}/model_${LOW_CHECKPOINT_STEP}.pt}"
TARGET_NAME="${TARGET_NAME:-publiccheckrollrew_${LOW_CHECKPOINT_STEP}.pt}"
TARGET_PATH="${PROJECT_DIR}/high-level/data/low_policy/${TARGET_NAME}"
CFG_FILE="${PROJECT_DIR}/high-level/data/cfg/b1z1_pickmulti.yaml"

if [[ ! -f "$LOW_CKPT" ]]; then
  echo "Missing low-level checkpoint: ${LOW_CKPT}" >&2
  exit 5
fi

mkdir -p "$(dirname "$TARGET_PATH")"
cp "$LOW_CKPT" "$TARGET_PATH"

activate_env

python - <<'PY' "$CFG_FILE" "$TARGET_NAME"
import sys
import yaml

cfg_file, target_name = sys.argv[1], sys.argv[2]
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)
cfg["env"]["low_policy_path"] = f"data/low_policy/{target_name}"
with open(cfg_file, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "Prepared teacher low policy:"
echo "  source: ${LOW_CKPT}"
echo "  target: ${TARGET_PATH}"
echo "  cfg: ${CFG_FILE}"
echo "Next: TEACHER_RUN_NAME='teacher-after-${LOW_RUN_NAME}' bash remote-run/remote/50_start_teacher_tmux.sh"
