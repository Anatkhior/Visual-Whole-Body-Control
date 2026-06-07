#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -z "${https_proxy:-}${HTTPS_PROXY:-}" ]] && command -v curl >/dev/null 2>&1; then
  for proxy in http://127.0.0.1:7890 http://127.0.0.1:7892; do
    if HTTPS_PROXY="$proxy" HTTP_PROXY="$proxy" curl -fsSIL --connect-timeout 4 --max-time 8 https://pypi.org/simple/pip/ >/dev/null 2>&1; then
      export http_proxy="$proxy" https_proxy="$proxy" HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
      echo "Using local proxy: ${proxy}"
      break
    fi
  done
fi

install_miniconda_if_needed
conda_profile="$(conda_sh)" || {
  echo "Cannot find conda. Install Miniconda first or expose conda in PATH." >&2
  exit 3
}
# shellcheck source=/dev/null
source "$conda_profile"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Conda env exists: ${CONDA_ENV_NAME}"
else
  conda create -y -n "$CONDA_ENV_NAME" "python=${PYTHON_VERSION}"
fi

conda activate "$CONDA_ENV_NAME"
python -V

python -m pip install --upgrade pip setuptools wheel ninja

# PyTorch 2.4.1+cu121 worked on the local machine with this project. If the
# remote driver is too old, this command will fail early.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

python -m pip install \
  numpy==1.23.5 \
  pydelatin==0.2.8 \
  tqdm \
  imageio-ffmpeg \
  opencv-python \
  wandb \
  torchinfo==1.8.0 \
  matplotlib \
  gym==0.26.2 \
  gymnasium==1.1.1 \
  tensorboard \
  pyyaml

python -m pip install -e "${PROJECT_DIR}/third_party/rsl_rl"
python -m pip install -e "${PROJECT_DIR}/third_party/skrl"
python -m pip install -e "${PROJECT_DIR}/low-level" --no-deps

echo "Environment install complete: ${CONDA_ENV_NAME}"
