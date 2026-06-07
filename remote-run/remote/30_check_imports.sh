#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_env
cd "${PROJECT_DIR}/high-level"

python - <<'PY'
import os
import isaacgym
import torch
from utils.config import load_cfg
from envs import B1Z1PickMulti
from utils.low_level_model import ActorCritic

cfg = load_cfg("data/cfg/b1z1_pickmulti.yaml")
policy_path = cfg["env"]["low_policy_path"]

print("REMOTE_HIGH_LEVEL_IMPORT_CHECK")
print("cwd", os.getcwd())
print("isaacgym", isaacgym.__file__)
print("torch", torch.__version__)
print("torch_cuda_available", torch.cuda.is_available())
print("torch_cuda_device_count", torch.cuda.device_count())
print("task", B1Z1PickMulti.__name__)
print("actor_critic", ActorCritic.__name__)
print("low_policy_path", policy_path)
print("low_policy_exists", os.path.exists(policy_path))
ckpt = torch.load(policy_path, map_location="cpu")
print("ckpt_iter", ckpt.get("iter"))
print("model_state_dict_keys", len(ckpt["model_state_dict"]))
print("cfg_num_envs", cfg["env"]["numEnvs"])
PY
