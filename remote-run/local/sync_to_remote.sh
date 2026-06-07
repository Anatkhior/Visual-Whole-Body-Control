#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/remote.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing ${ENV_FILE}. Copy remote.env.example to remote.env and fill remote settings." >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

: "${REMOTE_USER:?REMOTE_USER is required}"
: "${REMOTE_HOST:?REMOTE_HOST is required}"
: "${REMOTE_PORT:=22}"
: "${REMOTE_ROOT:?REMOTE_ROOT is required}"
: "${SSH_KEY:=}"
: "${LOW_POLICY_LOCAL_PATH:=}"
: "${RSYNC_EXTRA_ARGS:=}"

SSH_OPTS=(-p "$REMOTE_PORT")
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi
SSH_OPTS+=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

echo "Creating remote root: ${REMOTE}:${REMOTE_ROOT}"
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_ROOT'"

echo "Syncing project to ${REMOTE}:${REMOTE_ROOT}"
rsync -az --info=progress2 --delete \
  --delete-excluded \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.git/' \
  --exclude 'high-level/experiments-smoke/' \
  --exclude 'high-level/data/asset/b1z1-float/' \
  --exclude 'low-level/logs/videos/' \
  --exclude 'third_party/isaacgym/assets/' \
  --exclude 'third_party/isaacgym/docker/' \
  --exclude 'third_party/isaacgym/docs/' \
  --exclude '**/__pycache__/' \
  --exclude '**/*.pyc' \
  --exclude 'remote-run/local/remote.env' \
  ${RSYNC_EXTRA_ARGS} \
  "${LOCAL_ROOT}/Visual-Whole-Body-Control/" \
  "${REMOTE}:${REMOTE_ROOT}/Visual-Whole-Body-Control/"

rsync -az --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
  --exclude 'remote-run/local/remote.env' \
  --exclude 'local/remote.env' \
  "${LOCAL_ROOT}/agent-log" \
  "${LOCAL_ROOT}/remote-run" \
  "${REMOTE}:${REMOTE_ROOT}/"

low_policy_to_sync=""
low_policy_candidates=()
if [[ -n "$LOW_POLICY_LOCAL_PATH" ]]; then
  if [[ "$LOW_POLICY_LOCAL_PATH" = /* ]]; then
    low_policy_candidates+=("$LOW_POLICY_LOCAL_PATH")
  else
    low_policy_candidates+=("${LOCAL_ROOT}/${LOW_POLICY_LOCAL_PATH}")
  fi
fi
low_policy_candidates+=("${LOCAL_ROOT}/model_38000.pt" "${LOCAL_ROOT}/../model_38000.pt")

for candidate in "${low_policy_candidates[@]}"; do
  if [[ -f "$candidate" ]]; then
    candidate_dir="$(cd "$(dirname "$candidate")" && pwd)"
    low_policy_to_sync="${candidate_dir}/$(basename "$candidate")"
    break
  fi
done

if [[ -n "$low_policy_to_sync" ]]; then
  echo "Syncing low-level policy: ${low_policy_to_sync}"
  rsync -az --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
    "$low_policy_to_sync" \
    "${REMOTE}:${REMOTE_ROOT}/"
else
  echo "No local low-level policy checkpoint was synced." >&2
  echo "Set LOW_POLICY_LOCAL_PATH in remote.env if the remote run needs a checkpoint upload." >&2
fi

echo "Remote sync complete."
echo "Next:"
echo "ssh ${SSH_OPTS[*]} ${REMOTE}"
echo "cd ${REMOTE_ROOT}"
echo "bash remote-run/remote/00_probe_remote.sh"
