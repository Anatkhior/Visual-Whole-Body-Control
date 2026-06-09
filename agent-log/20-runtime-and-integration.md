# 运行与集成

README 中建议的环境为 `conda` 环境 `b1z1`，Python 版本为 `3.8`，原因是 Isaac Gym 要求 `python <= 3.8`。基础安装流程包含 PyTorch、Isaac Gym、`rsl_rl`、`skrl`、`low-level` editable 安装，以及 `numpy`、`pydelatin`、`tqdm`、`imageio-ffmpeg`、`opencv-python`、`wandb`。

早期会话只完成仓库同步与日志建立；2026-05-26 已开始做本机仿真运行修复。

2026-05-25 复现环境检查摘要：

- 本机 `nvidia-smi` 可见 NVIDIA GTX 1660 SUPER 6GB，驱动版本 `595.71.05`，显示 CUDA Version `13.2`。
- `conda` 存在，当前已有 `legged-nexus-py38` 环境，Python 为 `3.8.20`。
- 在 `legged-nexus-py38` 中，`isaacgym` 可以导入。
- 在 `legged-nexus-py38` 中，PyTorch 为 `2.4.1+cu121`，但 `torch.cuda.is_available()` 为 `False`，`torch.cuda.device_count()` 为 `0`，并出现 `Can't initialize NVML` 警告；当前不能直接用 CUDA 跑项目。
- `skrl` 未安装，会阻断 high-level teacher/student 训练与回放。
- `rsl_rl` 和 `legged_gym` 当前导入路径指向 `/home/hjr/projects/2-Nexus/Legged-Nexus`，不是本项目目录，建议新建干净环境或重新 editable 安装本项目内的依赖，避免路径污染。

同步方式摘要：

- 目标远端：`https://github.com/Anatkhior/Visual-Whole-Body-Control.git`
- 本地目录：`Visual-Whole-Body-Control/`
- 当前提交：`869104c31953718f30ad20675e5291fcb5c5ea23`
- 仓库状态：浅仓库，`origin/main` 指向当前提交。
- 恢复方式：`git clone` 传输失败后，下载同一提交的源码包 `/tmp/Visual-Whole-Body-Control-869104c.tar.gz`，解压到工作树，写入本地 Git blob 对象并重建索引。

运行前注意：

- Isaac Gym 安装路径位于 `third_party/isaacgym/python`。
- `third_party/` 内容较大，当前本地项目目录约 `1.9G`。
- 若后续需要完整 Git 历史，需在网络稳定时执行 `git fetch --unshallow`。
- 若要先做仿真复现，先解决 CUDA 可见性和 `skrl` 安装，再用本项目环境运行低层 `play.py` 或高层 `play_*.py`。

2026-05-26 修复摘要：

- 沙箱内看不到 `/dev/nvidia*`，会导致 `torch.cuda.is_available()` 为 `False`；沙箱外同一 `legged-nexus-py38` 环境中 CUDA 可见，`torch.cuda.is_available()` 为 `True`，`device_count=1`。
- 已安装缺失运行依赖：`torchinfo==1.8.0`、`pydelatin==0.2.8`、`gym==0.26.2`、`gymnasium==1.1.1`、`imageio-ffmpeg==0.5.1`。
- 已将 NumPy 从 `1.24.4` 降到 `1.23.5`，修复 Isaac Gym 旧代码中 `np.float` 被移除的问题。
- 运行时必须优先导入本项目自带 Isaac Gym：`Visual-Whole-Body-Control/third_party/isaacgym/python`。外部 `/home/hjr/Isaacgym/isaacgym/python` 缺少本项目需要的 `euler_from_quat`。
- 当前没有修改 conda 环境中已有的 `LeggedGym-Ex` editable 映射；用 `PYTHONPATH` 临时覆盖，避免影响 `/home/hjr/projects/2-Nexus/Legged-Nexus`。
- 直接调用环境 Python 时需要显式设置 `PATH=/home/hjr/miniconda3/envs/legged-nexus-py38/bin:...` 和 `LD_LIBRARY_PATH=/home/hjr/miniconda3/envs/legged-nexus-py38/lib`，否则 Isaac Gym 绑定可能找不到 `libpython3.8.so.1.0` 或 `ninja`。

2026-05-26 高层运行补充：

- high-level 也必须在沙箱外运行，原因与 low-level 相同：需要真实 `/dev/nvidia*`、可写的 `/home/hjr/.cache/torch_extensions/py38_cu121/gymtorch`，以及 Isaac Gym GPU PhysX。
- high-level 脚本中导入顺序很重要：必须先 `import isaacgym`，再导入会触发 PyTorch 的模块。
- 推荐继续使用显式环境变量运行 high-level：`PATH` 指向 `legged-nexus-py38/bin`，`LD_LIBRARY_PATH` 指向该 conda env 的 `lib`，`PYTHONPATH` 优先包含本项目 `third_party/isaacgym/python`、`high-level`、`third_party/skrl`。

2026-05-29 远端快速部署约定：

- 本地远端配置模板：`remote-run/local/remote.env.example`。复制为 `remote.env` 后填写 `REMOTE_USER`、`REMOTE_HOST`、`REMOTE_PORT`、`REMOTE_ROOT` 和可选 `SSH_KEY`。
- 同步脚本：`remote-run/local/sync_to_remote.sh`。会同步当前工作树的 `Visual-Whole-Body-Control/`、`model_38000.pt`、`agent-log/` 和 `remote-run/`，并排除 `.git`、`experiments-smoke`、视频和 Python cache。
- 远端环境默认名：`vbc-py38`。若远端没有 conda，`10_install_env.sh` 默认尝试安装 Miniconda 到 `~/miniconda3`。
- 远端运行时统一设置 `PYTHONPATH=${PROJECT_DIR}/third_party/isaacgym/python:${PROJECT_DIR}/high-level:${PROJECT_DIR}/third_party/skrl:${PROJECT_DIR}/third_party/rsl_rl:${PROJECT_DIR}/low-level`。
- 使用 `CUDA_VISIBLE_DEVICES` 选择物理 GPU 后，训练脚本内部统一用 `cuda:0`，避免单卡可见时设备编号错位。

2026-05-30 远端实测与启动：

- 远端可用信息：`ubuntu@<remote-host>:22`，工作目录 `/home/ubuntu/vbc-remote`，环境名 `vbc-py38`。
- 远端硬件体检结果：双 RTX 3090，驱动 `570.181`，CUDA `12.8`，磁盘剩余约 `153GB`。
- 远端环境安装已完成，`isaacgym`、`torch 2.4.1+cu121`、`rsl_rl`、`skrl`、`legged_gym` 均可导入。
- 网络安装走本机代理 `http://127.0.0.1:7890`；`remote-run/remote/10_install_env.sh` 已改为自动探测该代理，避免公网直连失败。
- `10_install_env.sh` 还补了 `matplotlib`，并把 `low-level` 的 editable 安装改为 `pip install -e ... --no-deps`，绕开 PyPI 中并不存在的 `isaacgym` 依赖查找。
- 项目同步采用打包 + 分片传输，远端合并后校验通过，避免直接目录 rsync 的速度问题。
- `20_prepare_project.sh` 已为 high-level 和 low-level 建好 `model_38000.pt` 软链接。
- 远端 smoke test `TEACHER_SMOKE_ENVS=10240`、`TEACHER_SMOKE_TIMESTEPS=24` 已通过，随后 teacher 长训已在 tmux 会话 `vbc_teacher_g0` 启动。

2026-06-06 student 视觉相机运行前提：

- 远端当前存在桌面图形会话：`DISPLAY=:0`，Xorg/gnome 进程运行中。
- Isaac Gym 自带 `multiple_camera_envs.py` 验证显示：SSH 默认不带 `DISPLAY` 时 camera 示例会 core dump；显式 `DISPLAY=:0` 时可正常创建 camera handles。
- student 视觉训练必须启用 camera sensors，且必须绑定 `DISPLAY=:0`。不要只设置 `--headless` 和 `CUDA_VISIBLE_DEVICES`；否则可能出现 camera handle `-1` 或 Isaac Gym 段错误。
- `remote-run/remote/70_start_student_tmux.sh` 已支持 `STUDENT_DISPLAY`，建议后续 student 启动统一设置 `STUDENT_DISPLAY=:0`。
2026-06-09 11:05:24 CST +0800 SSH key 恢复：

- 原远端私钥放在 `/tmp/vbc_remote_ed25519`，该路径会被系统或会话清理，不适合作为长期连接凭据。
- 已生成稳定本地 key：`/home/hjr/projects/2-Nexus/VBC/.secrets/ssh/vbc_remote_ed25519`。
- 已将对应公钥追加到远端 `~/.ssh/authorized_keys`，并用 `BatchMode=yes` 验证无密码 SSH，远端返回 `SSH_OK`。
- 私钥不在 `Visual-Whole-Body-Control` Git 仓库内，不应提交；私有外层 `remote-run/local/remote.env` 已把 `SSH_KEY` 更新为稳定路径。
