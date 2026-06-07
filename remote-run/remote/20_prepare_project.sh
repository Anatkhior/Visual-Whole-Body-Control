#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ ! -f "$LOW_POLICY_SRC" ]]; then
  echo "Missing low-level policy checkpoint: ${LOW_POLICY_SRC}" >&2
  exit 5
fi

LOW_POLICY_TARGET_NAME="${LOW_POLICY_TARGET_NAME:-$(basename "$LOW_POLICY_SRC")}"
LOW_POLICY_TARGET="${PROJECT_DIR}/high-level/data/low_policy/${LOW_POLICY_TARGET_NAME}"
LOW_POLICY_LOW_LEVEL_LINK="${PROJECT_DIR}/low-level/logs/b1z1-low/google_drive/${LOW_POLICY_TARGET_NAME}"

mkdir -p "${PROJECT_DIR}/high-level/data/low_policy"
if [[ "$(readlink -f "$LOW_POLICY_SRC")" != "$(readlink -m "$LOW_POLICY_TARGET")" ]]; then
  ln -sfn "$LOW_POLICY_SRC" "$LOW_POLICY_TARGET"
fi

mkdir -p "${PROJECT_DIR}/low-level/logs/b1z1-low/google_drive"
if [[ "$(readlink -f "$LOW_POLICY_SRC")" != "$(readlink -m "$LOW_POLICY_LOW_LEVEL_LINK")" ]]; then
  ln -sfn "$LOW_POLICY_SRC" "$LOW_POLICY_LOW_LEVEL_LINK"
fi

python_bin="python"
if command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
fi

"$python_bin" - <<'PY' "${PROJECT_DIR}/high-level/data/cfg/b1z1_pickmulti.yaml" "$LOW_POLICY_TARGET_NAME"
import sys
from pathlib import Path

path = Path(sys.argv[1])
target_name = sys.argv[2]
replacement = f'low_policy_path: "data/low_policy/{target_name}"'
text = path.read_text()
old = '/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt'
if old in text:
    text = text.replace(old, f"data/low_policy/{target_name}")
lines = []
replaced = False
for line in text.splitlines():
    if line.strip().startswith("low_policy_path:"):
        indent = line[:len(line) - len(line.lstrip())]
        lines.append(f"{indent}{replacement}")
        replaced = True
    else:
        lines.append(line)
if not replaced:
    raise SystemExit("Cannot find low_policy_path in config")
text = "\n".join(lines) + "\n"
path.write_text(text)
PY

echo "Prepared checkpoint links:"
ls -l "$LOW_POLICY_TARGET"
ls -l "$LOW_POLICY_LOW_LEVEL_LINK"
