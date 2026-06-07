# 运行与集成原始记录

<!-- source: session 2026-05-25, clone/download/recovery commands -->

工作区：`/home/hjr/projects/2-Nexus/VBC`

目标仓库：

- URL：`https://github.com/Anatkhior/Visual-Whole-Body-Control`
- Git remote：`https://github.com/Anatkhior/Visual-Whole-Body-Control.git`
- 本地目录：`Visual-Whole-Body-Control/`

同步尝试：

1. `git clone https://github.com/Anatkhior/Visual-Whole-Body-Control.git Visual-Whole-Body-Control`
   - 先在沙箱网络下失败：`Could not resolve host: github.com`
   - 授权网络后仍失败：`curl 56 GnuTLS recv error (-9)`、`fetch-pack: unexpected disconnect while reading sideband packet`

2. `git clone --depth 1 --single-branch https://github.com/Anatkhior/Visual-Whole-Body-Control.git Visual-Whole-Body-Control`
   - 失败原因同样是 TLS/RPC 传输中断。

3. `git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none --single-branch https://github.com/Anatkhior/Visual-Whole-Body-Control.git Visual-Whole-Body-Control`
   - 产生部分 `.git` 和少量工作树文件。
   - checkout 过程中出现缺失对象和 `index-pack` 失败，工作树不完整。

恢复步骤：

1. 读取部分克隆留下的当前提交：`869104c31953718f30ad20675e5291fcb5c5ea23`。
2. 下载同一提交源码包：
   `curl -L --retry 5 --retry-delay 2 --fail -o /tmp/Visual-Whole-Body-Control-869104c.tar.gz https://github.com/Anatkhior/Visual-Whole-Body-Control/archive/869104c31953718f30ad20675e5291fcb5c5ea23.tar.gz`
3. 完整遍历源码包：
   `tar -tzf /tmp/Visual-Whole-Body-Control-869104c.tar.gz > /tmp/Visual-Whole-Body-Control-869104c.tar.list`
4. 解压补齐工作树：
   `tar -xzf /tmp/Visual-Whole-Body-Control-869104c.tar.gz --strip-components=1 -C Visual-Whole-Body-Control`
5. 移走陈旧锁：
   `mv Visual-Whole-Body-Control/.git/index.lock /tmp/Visual-Whole-Body-Control.index.lock.stale`
6. 使用仓库内相对路径补写 blob 对象：
   `git -C Visual-Whole-Body-Control ls-tree -rz -r --full-tree HEAD | while IFS=$'\t' read -r -d '' meta path; do if [ -f "Visual-Whole-Body-Control/$path" ]; then git -C Visual-Whole-Body-Control hash-object -w -- "$path" >/dev/null; else printf '%s\0' "$path" >> /tmp/Visual-Whole-Body-Control-missing-files-2.zlist; fi; done`
7. 重建索引：
   `git -C Visual-Whole-Body-Control read-tree HEAD`

最终状态：

- `git -C Visual-Whole-Body-Control rev-parse HEAD` 输出 `869104c31953718f30ad20675e5291fcb5c5ea23`。
- `git -C Visual-Whole-Body-Control remote -v` 显示 fetch/push 均指向 `origin https://github.com/Anatkhior/Visual-Whole-Body-Control.git`。
- `git -C Visual-Whole-Body-Control rev-parse --is-shallow-repository` 输出 `true`。

<!-- source: session 2026-05-25, reproduction environment checks -->

复现环境只读检查：

- `nvidia-smi`：
  - GPU：NVIDIA GeForce GTX 1660 SUPER
  - 显存：6144 MiB
  - Driver Version：595.71.05
  - CUDA Version：13.2
- `which conda`：`/home/hjr/miniconda3/condabin/conda`
- `conda env list`：存在 `base` 与 `legged-nexus-py38`
- `conda run -n legged-nexus-py38 python --version`：`Python 3.8.20`
- `conda run -n legged-nexus-py38 python -c "import torch; ..."`：
  - `torch 2.4.1+cu121`
  - `torch_cuda 12.1`
  - `device_count 0`
  - `is_available False`
  - 警告：`Can't initialize NVML`
- `conda run -n legged-nexus-py38 python -c "import isaacgym; print('isaacgym ok')"`：成功。
- `conda run -n legged-nexus-py38 python -c "import skrl"`：失败，`ModuleNotFoundError: No module named 'skrl'`。
- `conda run -n legged-nexus-py38 python -c "import rsl_rl; print(rsl_rl.__file__)"`：输出 `/home/hjr/projects/2-Nexus/Legged-Nexus/rsl_rl/__init__.py`。
- `conda run -n legged-nexus-py38 python -c "import legged_gym; print(legged_gym.__file__)"`：输出 `/home/hjr/projects/2-Nexus/Legged-Nexus/legged_gym/__init__.py`。

解释：当前已有 Python 3.8 和 Isaac Gym，但 high-level 所需 `skrl` 缺失，且 `rsl_rl/legged_gym` 指向其他项目，PyTorch 无法看到 CUDA。建议新建干净环境或重新安装本项目依赖。

<!-- source: session 2026-05-26 09:09:37 CST +0800, simulation runtime repair -->

仿真运行修复记录：

- 沙箱内检查：
  - `ls -l /dev/nvidia*` 失败，显示没有 `/dev/nvidia*`。
  - `conda run -n legged-nexus-py38 python -c "import torch; ..."` 在沙箱内显示 `torch.cuda.is_available() == False`。
- 沙箱外检查：
  - 同一环境中 `torch 2.4.1+cu121`、`torch.version.cuda 12.1`、`torch.cuda.is_available() == True`、`torch.cuda.device_count() == 1`、设备名为 `NVIDIA GeForce GTX 1660 SUPER`。
  - 结论：CUDA 本身可用，仿真需要在真实终端或授权的沙箱外环境运行。
- 缺失依赖安装：
  - 执行：`conda run -n legged-nexus-py38 pip install torchinfo pydelatin gym gymnasium imageio-ffmpeg`
  - 安装后版本：`torchinfo 1.8.0`、`pydelatin 0.2.8`、`gym 0.26.2`、`gymnasium 1.1.1`、`imageio-ffmpeg 0.5.1`。
- NumPy 兼容修复：
  - `play.py` 初次启动报错：`AttributeError: module 'numpy' has no attribute 'float'`。
  - 原版本：`numpy 1.24.4`。
  - 执行：`conda run -n legged-nexus-py38 pip install numpy==1.23.5`
  - 修复后：`numpy.__version__ == 1.23.5`，`np.float` 可用但有弃用警告。
- 路径污染处理：
  - 当前环境已有 `LeggedGym-Ex 0.3.0` editable finder，把 `legged_gym` 和 `rsl_rl` 映射到 `/home/hjr/projects/2-Nexus/Legged-Nexus`。
  - 未卸载该包；运行 VBC 时通过 `PYTHONPATH` 让本项目路径优先。
  - 推荐 low-level 运行前缀：
    `PYTHONPATH=/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/isaacgym/python:/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level:/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/rsl_rl:/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/skrl`
  - 推荐 high-level 运行前缀：
    `PYTHONPATH=/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/isaacgym/python:/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level:/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/skrl`
- 直接调用环境 Python 时的额外要求：
  - `PATH` 需要包含 `/home/hjr/miniconda3/envs/legged-nexus-py38/bin`，否则加载 `gymtorch` 时可能报 `Ninja is required to load C++ extensions`。
  - `LD_LIBRARY_PATH` 需要包含 `/home/hjr/miniconda3/envs/legged-nexus-py38/lib`，否则可能报 `libpython3.8.so.1.0` 找不到。
<!-- source: session 2026-05-29 16:16:54 CST +0800, remote deployment preparation -->

远端快速部署准备：

- 新增目录：`/home/hjr/projects/2-Nexus/VBC/remote-run/`
- 本地配置模板：`remote-run/local/remote.env.example`
  - 必填：`REMOTE_USER`、`REMOTE_HOST`、`REMOTE_PORT`、`REMOTE_ROOT`
  - 可选：`SSH_KEY`、`RSYNC_EXTRA_ARGS`
- 本地同步脚本：`remote-run/local/sync_to_remote.sh`
  - 同步当前工作树 `Visual-Whole-Body-Control/`
  - 同步 `model_38000.pt`
  - 同步 `agent-log/`
  - 同步 `remote-run/`
  - 排除 `.git`、`high-level/experiments-smoke/`、`low-level/logs/videos/`、`__pycache__` 和 `.pyc`
- 本地取回脚本：`remote-run/local/fetch_results.sh`
  - 拉取远端 `remote-logs/`
  - 拉取远端 `vbc-results-*.tar.gz`
- SSH 安全脚本：`remote-run/local/generate_ssh_key.sh`
  - 默认生成 `~/.ssh/vbc_remote_ed25519`
  - 输出 public key，供用户加入远端 `~/.ssh/authorized_keys`

远端脚本：

- `00_probe_remote.sh`：检查主机、系统、GPU、显存、内存、磁盘、conda、tmux、rsync、gcc/g++。
- `10_install_env.sh`：创建/复用 `vbc-py38` conda 环境；若无 conda 且 `AUTO_INSTALL_MINICONDA=1`，尝试安装 Miniconda 到 `~/miniconda3`；安装 PyTorch `2.4.1+cu121`、NumPy `1.23.5`、`pydelatin`、`gym`、`gymnasium`、`wandb`、`tensorboard` 等，并 editable 安装 `rsl_rl`、`skrl`、`low-level`。
- `20_prepare_project.sh`：建立 low-level/high-level 到 `model_38000.pt` 的软链接，并确保 high-level 配置中的 `low_policy_path` 指向 `data/low_policy/model_38000.pt`。
- `30_check_imports.sh`：按正确顺序先导入 `isaacgym`，再导入 PyTorch、`B1Z1PickMulti`、`ActorCritic`，并读取 low-level checkpoint。
- `40_smoke_teacher.sh`：临时把 `data/cfg/b1z1_pickmulti.yaml` 的 `numEnvs` 改为 `TEACHER_SMOKE_ENVS`，默认 `10240`；运行 `train_multistate.py` 的 `TEACHER_SMOKE_TIMESTEPS` 短训练，默认 `24`；退出时恢复配置。
- `50_start_teacher_tmux.sh`：在 tmux 中启动 teacher 长训练，日志写到 `${REMOTE_ROOT}/remote-logs/${TEACHER_RUN_NAME}.teacher.log`。
- `60_monitor_teacher.sh`：查看 GPU、checkpoint 和 teacher 日志尾部。
- `70_start_student_tmux.sh`：在 `STUDENT_TEACHER_CKPT` 已设置时启动 student 训练。
- `80_collect_results.sh`：打包 teacher/student 目录、remote logs 和 agent-log。

远端 GPU 约定：

- 使用 `CUDA_VISIBLE_DEVICES=${TEACHER_GPU}` / `CUDA_VISIBLE_DEVICES=${STUDENT_GPU}` 绑定物理卡。
- 绑定后脚本内部传给项目的 `--rl_device` 和 `--sim_device` 固定为 `cuda:0`，避免单卡可见时 CUDA 编号错位。

<!-- source: session 2026-05-30 16:18:43 CST +0800, remote dual-3090 actual run -->

远端实际运行补记：

- 远端主机：`ubuntu@<remote-host>:22`
- 远端目录：`/home/ubuntu/vbc-remote`
- 远端环境：`vbc-py38`
- 远端硬件：双 RTX 3090，驱动 `570.181`，CUDA `12.8`
- 远端可用磁盘：约 `153GB`
- `remote-run/remote/10_install_env.sh` 的实际使用结果：
  - 代理自动探测到了 `http://127.0.0.1:7890`
  - `matplotlib` 已安装
  - `low-level` 采用 `pip install -e ... --no-deps`
- 同步完成后已执行 `20_prepare_project.sh`，建立 `model_38000.pt` 软链接。
- smoke 通过后，teacher 已进入 tmux 长训。
