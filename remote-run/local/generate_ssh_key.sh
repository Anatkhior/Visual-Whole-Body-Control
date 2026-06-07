#!/usr/bin/env bash
set -euo pipefail

KEY_PATH="${1:-$HOME/.ssh/vbc_remote_ed25519}"

mkdir -p "$(dirname "$KEY_PATH")"
chmod 700 "$(dirname "$KEY_PATH")"

if [[ -e "$KEY_PATH" ]]; then
  echo "SSH key already exists: $KEY_PATH"
else
  ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "vbc-remote-$(hostname)-$(date +%Y%m%d)"
fi

echo
echo "Public key to add on the remote host:"
cat "${KEY_PATH}.pub"
echo
echo "Private key path: $KEY_PATH"
