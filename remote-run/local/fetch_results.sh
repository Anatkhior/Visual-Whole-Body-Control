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

SSH_OPTS=(-p "$REMOTE_PORT")
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi
SSH_OPTS+=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
DEST="${LOCAL_ROOT}/remote-results"
mkdir -p "$DEST"

echo "Fetching remote logs and result tarballs to ${DEST}"
rsync -az --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
  "${REMOTE}:${REMOTE_ROOT}/remote-logs/" \
  "${DEST}/remote-logs/" || true

while IFS= read -r result_file; do
  [[ -n "$result_file" ]] || continue
  rsync -az --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
    "${REMOTE}:${result_file}" \
    "${DEST}/" || true
done < <(ssh "${SSH_OPTS[@]}" "$REMOTE" "find '$REMOTE_ROOT' -maxdepth 1 -type f -name 'vbc-results-*.tar.gz' -print")

echo "Fetch complete: ${DEST}"
