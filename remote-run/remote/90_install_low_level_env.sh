#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

LOW_CONDA_ENV_NAME="${LOW_CONDA_ENV_NAME:-vbc-low-cu113}"
LOW_PYTHON_VERSION="${LOW_PYTHON_VERSION:-3.8.18}"

if [[ -z "${https_proxy:-}${HTTPS_PROXY:-}" ]] && command -v curl >/dev/null 2>&1; then
  for proxy in http://127.0.0.1:7890 http://127.0.0.1:7892; do
    if HTTPS_PROXY="$proxy" HTTP_PROXY="$proxy" curl -fsSIL --connect-timeout 4 --max-time 8 https://pypi.org/simple/pip/ >/dev/null 2>&1; then
      export http_proxy="$proxy" https_proxy="$proxy" HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy"
      echo "Using local proxy: ${proxy}"
      break
    fi
  done
fi

conda_profile="$(conda_sh)" || {
  echo "Cannot find conda. Install Miniconda/Anaconda first." >&2
  exit 3
}
# shellcheck source=/dev/null
source "$conda_profile"

if conda env list | awk '{print $1}' | grep -qx "$LOW_CONDA_ENV_NAME"; then
  echo "Conda env exists: ${LOW_CONDA_ENV_NAME}"
else
  conda create -y -n "$LOW_CONDA_ENV_NAME" "python=${LOW_PYTHON_VERSION}"
fi

conda activate "$LOW_CONDA_ENV_NAME"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

python -V
python -m pip install --upgrade pip==23.3.1 setuptools==68.2.2 wheel==0.41.2 ninja==1.11.1.1

python -m pip install --index-url https://download.pytorch.org/whl/cu113 \
  torch==1.10.0+cu113 \
  torchvision==0.11.1+cu113 \
  torchaudio==0.10.0+cu113

python -m pip install \
  appdirs==1.4.4 \
  certifi==2023.11.17 \
  charset-normalizer==3.3.2 \
  click==8.1.7 \
  contourpy==1.1.1 \
  cycler==0.12.1 \
  docker-pycreds==0.4.0 \
  fonttools==4.47.0 \
  gitdb==4.0.11 \
  gitpython==3.1.41 \
  idna==3.6 \
  imageio==2.33.1 \
  imageio-ffmpeg \
  importlib-resources==6.1.1 \
  kiwisolver==1.4.5 \
  matplotlib==3.7.4 \
  numpy==1.23.5 \
  packaging==23.2 \
  pillow==10.2.0 \
  protobuf==4.25.2 \
  psutil==5.9.7 \
  pydelatin==0.2.7 \
  pyparsing==3.1.1 \
  python-dateutil==2.8.2 \
  pyyaml==6.0.1 \
  requests==2.31.0 \
  scipy==1.10.1 \
  sentry-sdk==1.39.2 \
  setproctitle==1.3.3 \
  six==1.16.0 \
  smmap==5.0.1 \
  torchinfo==1.8.0 \
  tqdm==4.66.1 \
  typing-extensions==4.9.0 \
  urllib3==2.1.0 \
  wandb==0.16.2 \
  zipp==3.17.0 \
  tensorboard \
  gym==0.26.2

python -m pip install -e "${PROJECT_DIR}/third_party/isaacgym/python"
python -m pip install -e "${PROJECT_DIR}/third_party/rsl_rl"
python -m pip install -e "${PROJECT_DIR}/low-level" --no-deps

python - <<'PY'
import isaacgym
import torch
import numpy
import wandb
print("python_env_ok")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("wandb", wandb.__version__)
PY

echo "Low-level author-like environment ready: ${LOW_CONDA_ENV_NAME}"
