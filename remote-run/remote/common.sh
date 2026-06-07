#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="${PROJECT_DIR:-${REMOTE_ROOT}/Visual-Whole-Body-Control}"
LOW_POLICY_SRC="${LOW_POLICY_SRC:-${REMOTE_ROOT}/model_38000.pt}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vbc-py38}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
AUTO_INSTALL_MINICONDA="${AUTO_INSTALL_MINICONDA:-1}"

TEACHER_EXPERIMENT_DIR="${TEACHER_EXPERIMENT_DIR:-b1-pick-multi-teacher}"
TEACHER_RUN_NAME="${TEACHER_RUN_NAME:-teacher-$(date +%Y%m%d-%H%M%S)}"
TEACHER_TIMESTEPS="${TEACHER_TIMESTEPS:-60000}"
TEACHER_GPU="${TEACHER_GPU:-0}"
TEACHER_SMOKE_TIMESTEPS="${TEACHER_SMOKE_TIMESTEPS:-24}"
TEACHER_SMOKE_ENVS="${TEACHER_SMOKE_ENVS:-10240}"
VBC_USE_WANDB="${VBC_USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-b1-pick-multi-teacher}"

STUDENT_EXPERIMENT_DIR="${STUDENT_EXPERIMENT_DIR:-b1-pick-multi-stu}"
STUDENT_RUN_NAME="${STUDENT_RUN_NAME:-student-$(date +%Y%m%d-%H%M%S)}"
STUDENT_TIMESTEPS="${STUDENT_TIMESTEPS:-60000}"
STUDENT_GPU="${STUDENT_GPU:-1}"
STUDENT_TEACHER_CKPT="${STUDENT_TEACHER_CKPT:-}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-}"

conda_sh() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_base
    conda_base="$(dirname "$(dirname "$CONDA_EXE")")"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      echo "${conda_base}/etc/profile.d/conda.sh"
      return
    fi
  fi
  for path in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return
    fi
  done
  return 1
}

install_miniconda_if_needed() {
  if conda_sh >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$AUTO_INSTALL_MINICONDA" != "1" ]]; then
    echo "Conda not found and AUTO_INSTALL_MINICONDA!=1." >&2
    return 1
  fi
  local installer="${REMOTE_ROOT}/Miniconda3-latest-Linux-x86_64.sh"
  local prefix="${HOME}/miniconda3"
  echo "Conda not found. Installing Miniconda to ${prefix}"
  if [[ ! -f "$installer" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    else
      echo "Neither curl nor wget is available for Miniconda download." >&2
      return 1
    fi
  fi
  bash "$installer" -b -p "$prefix"
}

activate_env() {
  local conda_profile
  conda_profile="$(conda_sh)" || {
    echo "Cannot find conda.sh. Install Miniconda/Anaconda or set CONDA_EXE." >&2
    exit 3
  }
  # shellcheck source=/dev/null
  source "$conda_profile"
  conda activate "$CONDA_ENV_NAME"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export PYTHONPATH="${PROJECT_DIR}/third_party/isaacgym/python:${PROJECT_DIR}/high-level:${PROJECT_DIR}/third_party/skrl:${PROJECT_DIR}/third_party/rsl_rl:${PROJECT_DIR}/low-level:${PYTHONPATH:-}"
}

teacher_args() {
  local use_wandb=()
  if [[ "$VBC_USE_WANDB" == "1" ]]; then
    use_wandb=(--wandb --wandb_project "$WANDB_PROJECT")
  fi
  printf '%q ' \
    --rl_device "cuda:0" \
    --sim_device "cuda:0" \
    --timesteps "$TEACHER_TIMESTEPS" \
    --headless \
    --task B1Z1PickMulti \
    --experiment_dir "$TEACHER_EXPERIMENT_DIR" \
    --wandb_name "$TEACHER_RUN_NAME" \
    --roboinfo \
    --observe_gait_commands \
    --small_value_set_zero \
    --rand_control \
    --stop_pick \
    "${use_wandb[@]}"
}

student_args() {
  if [[ -z "$STUDENT_TEACHER_CKPT" ]]; then
    echo "STUDENT_TEACHER_CKPT is required for student training." >&2
    exit 4
  fi
  local use_wandb=()
  local checkpoint_args=()
  if [[ "$VBC_USE_WANDB" == "1" ]]; then
    use_wandb=(--wandb --wandb_project "${WANDB_PROJECT}-student")
  fi
  if [[ -n "$STUDENT_CHECKPOINT" ]]; then
    checkpoint_args=(--checkpoint "$STUDENT_CHECKPOINT")
  fi
  printf '%q ' \
    --rl_device "cuda:0" \
    --sim_device "cuda:0" \
    --timesteps "$STUDENT_TIMESTEPS" \
    --headless \
    --task B1Z1PickMulti \
    --experiment_dir "$STUDENT_EXPERIMENT_DIR" \
    --wandb_name "$STUDENT_RUN_NAME" \
    --teacher_ckpt_path "$STUDENT_TEACHER_CKPT" \
    --roboinfo \
    --observe_gait_commands \
    --small_value_set_zero \
    --rand_control \
    --stop_pick \
    "${checkpoint_args[@]}" \
    "${use_wandb[@]}"
}
