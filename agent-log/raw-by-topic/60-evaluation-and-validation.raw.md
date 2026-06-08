# 评估与验证原始记录

<!-- source: session 2026-05-25, verification commands -->

已执行验证命令和关键输出：

- `tar -tzf /tmp/Visual-Whole-Body-Control-869104c.tar.gz | head -40`
  - 能列出顶层目录 `Visual-Whole-Body-Control-869104c31953718f30ad20675e5291fcb5c5ea23/` 及 README、`high-level/` 等文件。

- `tar -tzf /tmp/Visual-Whole-Body-Control-869104c.tar.gz > /tmp/Visual-Whole-Body-Control-869104c.tar.list`
  - 退出码 `0`。

- `wc -l /tmp/Visual-Whole-Body-Control-869104c.tar.list`
  - 输出 `2431 /tmp/Visual-Whole-Body-Control-869104c.tar.list`。

- `git -C Visual-Whole-Body-Control read-tree HEAD`
  - 最终退出码 `0`。

- `git -C Visual-Whole-Body-Control status --short`
  - 空输出。

- `git -C Visual-Whole-Body-Control fsck --no-dangling`
  - 空输出。

- `git -C Visual-Whole-Body-Control ls-files | wc -l`
  - 输出 `2043`。

- `find Visual-Whole-Body-Control -type f | wc -l`
  - 输出 `3496`，包含 `.git` 内部文件和工作树文件。

- `git -C Visual-Whole-Body-Control log -1 --oneline --decorate`
  - 输出 `869104c (grafted, HEAD -> main, origin/main, origin/HEAD) add ack`。

<!-- source: session 2026-05-25, low-level checkpoint and environment verification -->

低层 checkpoint 验证：

- `ls -lh /home/hjr/projects/2-Nexus/VBC/model_38000.pt`
  - 输出显示文件存在，大小约 `2.0M`。
- `file /home/hjr/projects/2-Nexus/VBC/model_38000.pt`
  - 输出：`Zip archive data, at least v0.0 to extract, compression method=store`。
  - 解释：PyTorch 新 checkpoint 格式本身使用 zip 容器，不需要当作普通压缩包手动解压。
- `sha256sum /home/hjr/projects/2-Nexus/VBC/model_38000.pt`
  - 输出：`a39f77e5c91298a6e3353664f55caf7afcc2cda4c0fd82ce6659c9fff21a6e23`
- `conda run -n legged-nexus-py38 python -c "import torch; p='/home/hjr/projects/2-Nexus/VBC/model_38000.pt'; ckpt=torch.load(p,map_location='cpu'); ..."`
  - 输出类型：`<class 'dict'>`
  - 顶层键：`['infos', 'iter', 'model_state_dict', 'optimizer_state_dict']`
  - `model_state_dict_keys 41`
  - 示例键：`std`、`actor.priv_encoder.0.weight`、`actor.priv_encoder.0.bias`、`actor.priv_encoder.2.weight`、`actor.priv_encoder.2.bias`、`actor.history_encoder.encoder.0.weight`、`actor.history_encoder.encoder.0.bias`、`actor.history_encoder.conv_layers.0.weight`
  - `iter 38000`

环境验证要点：

- `isaacgym` 可导入。
- `skrl` 不可导入。
- PyTorch 当前无法看到 CUDA，`torch.cuda.is_available()` 为 `False`。
- `legged_gym` 和 `rsl_rl` 当前导入路径指向 `Legged-Nexus`，不是本项目。

<!-- source: session 2026-05-26 09:09:37 CST +0800, repaired runtime verification -->

修复后验证命令和关键输出：

- CUDA 沙箱外检查：
  - 命令核心：`python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"`
  - 输出：`2.4.1+cu121`、`True`、`1`
- NumPy 兼容检查：
  - 命令核心：`python -c "import numpy as np; print(np.__version__); print(np.float)"`
  - 输出：`1.23.5`、`<class 'float'>`，伴随弃用警告。
- 本项目路径导入检查：
  - `isaacgym`：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/isaacgym/python/isaacgym/__init__.py`
  - `legged_gym`：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/legged_gym/__init__.py`
  - `rsl_rl`：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/rsl_rl/rsl_rl/__init__.py`
  - `skrl`：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/skrl/skrl/__init__.py`
- 低层 `play.py` smoke test 关键输出：
  - `PyTorch version 2.4.1+cu121`
  - `Device count 1`
  - `+++ Using GPU PhysX`
  - `Physics Device: cuda:0`
  - `GPU Pipeline: enabled`
  - `num_actions: 18`
  - `num_dofs: 19`
  - `root_states shape: torch.Size([1, 13])`
  - `Actor MLP: Actor(...)`
  - `Loading model from: /home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt`
  - 退出码为 `124`，原因是外层 `timeout` 主动截停。
- high-level 配置与 checkpoint 验证：
  - `low_policy_path data/low_policy/model_38000.pt`
  - `exists True`
  - checkpoint 键：`['infos', 'iter', 'model_state_dict', 'optimizer_state_dict']`
  - `iter 38000`
  - `model_state_dict_keys 41`
- high-level 导入验证：
  - `isaacgym` 指向本项目 `third_party/isaacgym`
  - `skrl` 指向本项目 `third_party/skrl`
  - `cfg_low_policy_path data/low_policy/model_38000.pt`
  - `B1Z1PickMulti B1Z1PickMulti`
  - `ActorCritic ActorCritic`
- 剩余环境警告：
  - `pip check` 报 `warp-sensor 0.2.0` 缺少 `proto-plus`、`pyopengl`、`pyqt5`、`pyzmq`、`typing`、`vizer`、`zmq`。
  - 该包不在当前 VBC low-level/high-level smoke test 直接路径上，暂不作为当前阻断项。

<!-- source: session 2026-05-26 09:42:41 CST +0800, longer low-level headless replay -->

较长低层 headless 数值回放结果：

- 回放设置：
  - 项目目录：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control`
  - 低层权重：`Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt`
  - checkpoint：`38000`
  - 设备：`cuda:0`，GPU PhysX
  - 步数：`1000`
  - 仿真时间：`20.0` 秒
- 关键输出：
  - `MOTION_CHECK_RESULT`
  - `steps`: `1000`
  - `sim_time_s`: `20.0`
  - `wall_time_s`: `15.062`
  - `checkpoint`: `38000`
  - `start_pos`: `[35.89782, 4.04187, 0.49611]`
  - `final_pos`: `[34.83368, 1.85176, 0.53489]`
  - `final_xy_displacement_m`: `2.43494`
  - `approx_xy_path_m`: `9.34779`
  - `root_z_min_m`: `0.48974`
  - `root_z_max_m`: `0.55614`
  - `mean_abs_action`: `4.82431`
  - `max_abs_action`: `18.82501`
  - `mean_reward`: `0.05604`
  - `mean_arm_reward`: `0.00679`
  - `done_count`: `1`
  - `command_mean`: `[0.32125, 0.0, 0.15175]`
  - `command_abs_mean`: `[0.32125, 0.0, 0.46234]`
  - `base_lin_vel_mean`: `[0.37739, -0.07561, 0.04544]`
  - `base_lin_vel_abs_mean`: `[0.37802, 0.11921, 0.07779]`
- 判读：
  - base position / velocity 在 step `200/400/600/800/1000` 持续变化。
  - 平面位移和路径长度说明策略有实际运动输出。
  - 根节点高度区间约 `0.49m` 到 `0.56m`，没有出现立刻倒地或坍塌的证据。
  - `done_count=1` 表示过程中发生过一次 episode reset 或 done，不能把这次验证解释为完整任务成功。

<!-- source: session 2026-05-26 10:28:28 CST +0800, rerun longer low-level headless replay -->

重新运行较长低层 headless 数值回放，用于避免只依赖上一轮交接摘要：

- 临时脚本：`/tmp/vbc_low_level_motion_check.py`
- 命令核心：
  - `python /tmp/vbc_low_level_motion_check.py --exptid google_drive --task b1z1 --proj_name b1z1-low --checkpoint 38000 --observe_gait_commands --flat_terrain --sim_device cuda:0 --rl_device cuda:0 --headless`
  - 运行前显式设置 `PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`，确保使用 `legged-nexus-py38` 环境和本项目 `third_party/isaacgym/python`、`low-level`、`third_party/rsl_rl`、`third_party/skrl`。
- 初始化关键输出：
  - `PyTorch version 2.4.1+cu121`
  - `Device count 1`
  - `+++ Using GPU PhysX`
  - `Physics Device: cuda:0`
  - `GPU Pipeline: enabled`
  - `Loading model from: /home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt`
- 进度输出：
  - step `200`：`pos=[37.51637, 4.19643, 0.53785]`，`base_lin_vel=[0.03321, -0.13939, 0.10546]`，`commands=[0.03462, 0.0, -0.56321]`，`done_count=0`
  - step `400`：`pos=[37.41567, 4.00654, 0.54178]`，`base_lin_vel=[-0.00042, -0.06726, -0.0113]`，`commands=[0.0, 0.0, -0.0]`，`done_count=0`
  - step `600`：`pos=[36.35179, 4.62214, 0.54242]`，`base_lin_vel=[0.45735, 0.16999, 0.06731]`，`commands=[0.39222, 0.0, 0.93505]`，`done_count=1`
  - step `800`：`pos=[34.7157, 4.0517, 0.53867]`，`base_lin_vel=[0.6018, 0.11409, 0.05902]`，`commands=[0.55499, 0.0, 0.74259]`，`done_count=1`
  - step `1000`：`pos=[34.79231, 1.8524, 0.53496]`，`base_lin_vel=[0.57128, 0.11144, 0.09784]`，`commands=[0.51151, 0.0, 0.76999]`，`done_count=1`
- 最终 `MOTION_CHECK_RESULT`：
  - `steps`: `1000`
  - `sim_time_s`: `20.0`
  - `wall_time_s`: `14.95`
  - `checkpoint`: `38000`
  - `start_pos`: `[35.89782, 4.04187, 0.49611]`
  - `final_pos`: `[34.79231, 1.8524, 0.53496]`
  - `final_xy_displacement_m`: `2.45274`
  - `approx_xy_path_m`: `9.32541`
  - `root_z_min_m`: `0.48385`
  - `root_z_max_m`: `0.55849`
  - `mean_abs_action`: `0.44452`
  - `max_abs_action`: `3.52549`
  - `mean_reward`: `0.05598`
  - `mean_arm_reward`: `0.00678`
  - `done_count`: `1`
  - `command_mean`: `[0.32125, 0.0, 0.15175]`
  - `command_abs_mean`: `[0.32125, 0.0, 0.46234]`
  - `base_lin_vel_mean`: `[0.37497, -0.0775, 0.0455]`
  - `base_lin_vel_abs_mean`: `[0.37574, 0.12179, 0.0791]`
- 退出状态：
  - 进程退出码 `0`。
  - 末尾有 `torch.load` 的 `FutureWarning`，不影响本次 checkpoint 加载和回放完成。

<!-- source: session 2026-05-26 11:06:53 CST +0800, fixed low-level headless video recording verification -->

低层 headless 视频录制修复后验证：

- 运行命令核心：
  - `python legged_gym/scripts/play.py --exptid google_drive --task b1z1 --proj_name b1z1-low --checkpoint 38000 --observe_gait_commands --flat_terrain --sim_device cuda:0 --rl_device cuda:0 --headless --record_video`
  - 运行前显式设置 `PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`，确保使用 `legged-nexus-py38` 和本项目 `third_party/isaacgym/python`、`low-level`、`third_party/rsl_rl`、`third_party/skrl`。
- 运行输出关键点：
  - `PyTorch version 2.4.1+cu121`
  - `Device count 1`
  - `+++ Using GPU PhysX`
  - `Physics Device: cuda:0`
  - `GPU Pipeline: enabled`
  - `Loading model from: /home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt`
  - `Recording video to: /home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/videos/google_drive/google_drive-0-38000.mp4`
  - 退出码：`0`
- 视频文件：
  - 路径：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/videos/google_drive/google_drive-0-38000.mp4`
  - `ls -lh`：约 `654K`
  - `stat`：`669454` bytes
  - `file`：`ISO Media, MP4 Base Media v1 [ISO 14496-12:2003]`
- `imageio`/ffmpeg 解码验证：
  - `codec`: `h264`
  - `pix_fmt`: `yuv420p`
  - `fps`: `25.0`
  - `source_size`: `(720, 480)`
  - `duration`: `10.0`
  - `frames`: `250`
  - `shapes`: `[(480, 720, 3)]`
- 内容变化验证：
  - `sample_indices`: `[0, 25, 50, 75, 100, 125, 150, 175, 200, 225]`
  - `sample_frame_abs_diffs`: `[15.305, 13.344, 9.056, 5.046, 6.379, 6.374, 2.188, 1.539, 1.643]`
  - `sample_max_diffs`: `[222, 211, 211, 222, 213, 213, 221, 209, 212]`
  - `sample_changed_pixel_fracs_gt10`: `[0.19425, 0.17025, 0.11549, 0.06892, 0.07868, 0.08119, 0.03568, 0.02575, 0.02865]`
  - `first_frame_std`: `44.608`
  - `first_last_abs_diff`: `14.505`
  - `first_last_changed_pixel_frac_gt10`: `0.17102`
- 判读：
  - 该 `.mp4` 是可解码视频容器，不是空文件。
  - 帧数、分辨率和 duration 与 `play.py` 录制逻辑一致：`record_video` 时 `traj_length=int(env.max_episode_length)`，`render_record()` 每 2 step 返回一次帧，writer fps `25`。
  - 抽样帧差和变化像素比例足以排除“完全静态帧/损坏容器”的问题。

<!-- source: session 2026-05-26 11:44:45 CST +0800, high-level teacher import and smoke tests -->

高层导入/config 检查：

- 运行环境：
  - `PATH=/home/hjr/miniconda3/envs/legged-nexus-py38/bin:...`
  - `LD_LIBRARY_PATH=/home/hjr/miniconda3/envs/legged-nexus-py38/lib`
  - `PYTHONPATH=/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/isaacgym/python:/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level:/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/skrl`
  - 工作目录：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level`
- 输出关键点：
  - `HIGH_LEVEL_IMPORT_CHECK`
  - `isaacgym /home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/third_party/isaacgym/python/isaacgym/__init__.py`
  - `torch_cuda_available True`
  - `torch_cuda_device_count 1`
  - `task B1Z1PickMulti`
  - `actor_critic ActorCritic`
  - `low_policy_path data/low_policy/model_38000.pt`
  - `low_policy_exists True`
  - `ckpt_iter 38000`
  - `model_state_dict_keys 41`
  - `cfg_num_envs 10240`

标准高层入口 `24` timestep 诊断：

- 命令核心：
  - `python train_multistate.py --rl_device cuda:0 --sim_device cuda:0 --timesteps 24 --headless --task B1Z1PickMulti --experiment_dir experiments-smoke --wandb_name smoke-high-level-teacher --debug --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`
- 运行输出关键点：
  - `Device count 1`
  - `+++ Using GPU PhysX`
  - `Using config file: data/cfg/b1z1_pickmulti.yaml`
  - `Low level pretrained policy loaded!`
  - tqdm 从 `0/24` 运行到 `23/24` 附近后进入 PPO update。
- 失败点：
  - `torch.OutOfMemoryError: CUDA out of memory`
  - 位置：`third_party/skrl/skrl/agents/torch/ppo/ppo.py` 的 `_update()`，进入 `self.optimizer.step()` 后在 PyTorch Adam foreach sqrt 中分配显存失败。
  - 报错显存：GPU 总容量约 `5.61 GiB`，仅剩 `1.75 MiB`，进程已占用约 `5.15 GiB`。
- 判读：
  - 该失败证明 34 env 标准 debug 配置在本机进入 PPO 更新时显存不足。
  - 它不是低层 checkpoint 缺失、导入错误或环境 reset 失败。

高层最小环境 `numEnvs=4` reset/step 验证：

- 临时脚本使用内联 Python，不写入项目文件。
- 配置覆盖：
  - `cfg['env']['numEnvs'] = 4`
  - `cfg['env']['maxEpisodeLength'] = 50`
  - `cfg['env']['wandb'] = False`
  - `cfg['env']['obj_move_prob'] = 0.0`
- 首次失败：
  - 先导入 `torch` 再导入 `isaacgym`，触发 Isaac Gym 导入顺序错误。
- 第二次失败：
  - `numEnvs=4` 时 `_reset_envs()` 的分类统计索引为空且 dtype 不适合作为 tensor index。
- 修复后重新运行，退出码 `0`。关键输出：
  - `HIGH_LEVEL_MINIMAL_ENV_SMOKE`
  - `numEnvs_override 4`
  - `low_policy_exists True`
  - `torch_cuda_available True`
  - `torch_cuda_device_count 1`
  - `Low level pretrained policy loaded!`
  - `env_num_envs 4`
  - `env_num_actions 9`
  - `env_num_obs 1094`
  - `obs_shape_after_reset (4, 1094)`
  - `low_level_policy_loaded method`
  - `step 1 obs_shape (4, 1094) reward_mean 0.006729 reset_sum 2 root0 [-1.9675, -0.1986, 0.5575] cube0 [0.1154, 0.0925, 0.2443]`
  - `step 4 obs_shape (4, 1094) reward_mean 0.038936 reset_sum 2 root0 [-1.9748, -0.1776, 0.5514] cube0 [0.1154, 0.0925, 0.2443]`
  - `step 8 obs_shape (4, 1094) reward_mean 0.015433 reset_sum 2 root0 [-1.9654, -0.1983, 0.5501] cube0 [0.1154, 0.0925, 0.2443]`
  - `total_reward_mean_sum 0.194517`
  - `total_reset_sum 16`
  - `cuda_mem_allocated_mb_after_steps 9.03`
  - `cuda_mem_reserved_mb_after_steps 24.0`

标准高层入口 `8` timestep 短 rollout：

- 命令核心：
  - `python train_multistate.py --rl_device cuda:0 --sim_device cuda:0 --timesteps 8 --headless --task B1Z1PickMulti --experiment_dir experiments-smoke --wandb_name smoke-high-level-teacher-rollout8 --debug --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`
- 输出关键点：
  - `Device count 1`
  - `+++ Using GPU PhysX`
  - `Using config file: data/cfg/b1z1_pickmulti.yaml`
  - `Low level pretrained policy loaded!`
  - tqdm 完成 `8/8`
  - `skrl:INFO Closing environment`
  - `skrl:INFO Environment closed`
- 退出码：`0`。

<!-- source: session 2026-05-30 16:18:43 CST +0800, remote teacher smoke validation -->

远端 teacher smoke 验证原始记录：

- smoke 命令：
  - `ssh -p 22 -i <ssh-key-path> ubuntu@<remote-host> "cd /home/ubuntu/vbc-remote && TEACHER_SMOKE_ENVS=10240 TEACHER_SMOKE_TIMESTEPS=24 bash remote-run/remote/40_smoke_teacher.sh"`
- 远端日志：
  - `/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260530-161843.log`
- 开头输出：
  - `REMOTE_TEACHER_SMOKE num_envs=10240 timesteps=24`
  - `Importing module 'gym_38' ...`
  - `+++ Using GPU PhysX`
  - `Physics Device: cuda:0`
  - `GPU Pipeline: enabled`
- 中段输出：
  - `Low level pretrained policy loaded!`
  - `Total success rate 0.0`
  - 对象计数持续增长，未见 OOM
  - tqdm 从 `0/24` 跑到 `24/24`
- 末尾输出：
  - `skrl:INFO Closing environment`
  - `Teacher smoke test finished.`
- 退出码：
  - `0`

远端 teacher 长训启动原始记录：

- 启动命令：
  - `ssh -p 22 -i <ssh-key-path> ubuntu@<remote-host> "cd /home/ubuntu/vbc-remote && bash remote-run/remote/50_start_teacher_tmux.sh"`
- 输出：
  - `Started teacher training in tmux session: vbc_teacher_g0`
  - `Log: /home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`
  - `Attach: tmux attach -t vbc_teacher_g0`
  - `Monitor: bash remote-run/remote/60_monitor_teacher.sh`

<!-- source: session 2026-05-31 12:56:19 CST +0800, remote teacher completion validation -->

远端 teacher 完成验证原始记录：

- 查询时间：
  - `2026-05-31 12:56:19 CST +0800`
- `tmux ls`：
  - 未列出 `vbc_teacher_g0`，说明训练命令已结束。
- 日志进度：
  - `60000/60000 [19:52:40<00:00, 1.19s/it]`
- 日志尾部：
  - `Teacher training exited at 2026-05-31 12:24:41 CST +0800`
- checkpoint 文件：
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/agent_60000.pt`，约 `20M`
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/best_agent.pt`，约 `20M`
- checkpoint 数量与总大小：
  - `find ... -name "agent_*.pt" | wc -l` 输出 `120`
  - `du -sh checkpoints` 输出 `2.4G`

<!-- source: session 2026-05-31 13:17:14-16:26:39 CST +0800, teacher eval validation and artifact sync -->

teacher eval 验证原始记录：

- `agent_60000.pt`：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-agent60000-20260531-131714.log`
  - 运行到约 `1508/50000` 后手动中断；`trainer.eval()` 未按 `--timesteps 3000` 截断，而是显示 `50000` 目标。
  - 中断前日志尾部显示多次 `Total success rate 0.0`。
  - 退出码：`130`，原因是手动停止长时间 eval。
- `best_agent.pt`：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-best-agent-20260531-132628.log`
  - 退出码：`1`
  - 错误：`ValueError: invalid literal for int() with base 10: 'agent'`
- `agent_58000.pt`：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-agent58000-20260531-132711.log`
  - 外层 `timeout 500s` 结束，退出码：`124`
  - 结束前进度约 `1687/50000`
  - 日志尾部显示多次 `Total success rate 0.0`
- 本地同步验证：
  - 本地目录：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-20260530-162617/`
  - `agent_60000.pt` 大小 `20622806`
  - `agent_58000.pt` 大小 `20622806`
  - `best_agent.pt` 大小 `20622696`
  - `torch.load` 顶层键均包括 `optimizer`、`policy`、`state_preprocessor`、`value`、`value_preprocessor`

<!-- source: session 2026-05-31 17:02:46 CST +0800, teacher probe validation and TensorBoard summary -->

teacher probe 验证原始记录：

- 临时 probe 脚本：
  - 本地：`/tmp/vbc_teacher_probe.py`
  - 远端：`/tmp/vbc_teacher_probe.py`
  - 作用：复用 `train_multistate.get_trainer(is_eval=True)` 创建 34 env eval，手动循环固定 steps，记录窗口成功数、episode 数、最大物体抬升、`lifted_object`、末端-物体最小距离和动作统计。
- `agent_60000.pt` 原 eval 起点：
  - checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/agent_60000.pt`
  - steps：`400`
  - `new_episodes=255`
  - `new_success=0`
  - `window_success_rate=0.0`
  - `max_lifted_object_count=0`
  - `max_curr_height=0.07588426768779755`
  - `min_curr_dist=0.03979865089058876`
- `agent_60000.pt` 训练起点 `x=-2.0`：
  - steps：`300`
  - `new_episodes=185`
  - `new_success=0`
  - `window_success_rate=0.0`
  - `max_lifted_object_count=0`
  - `max_curr_height=0.051837921142578125`
  - `min_curr_dist=0.05122927203774452`
- 真正 `best_agent.pt`：
  - 临时链接：`/tmp/best_58000.pt`
  - steps：`400`
  - `new_episodes=266`
  - `new_success=0`
  - `window_success_rate=0.0`
  - `max_lifted_object_count=0`
  - `max_curr_height=0.10852640867233276`
  - `min_curr_dist=0.019195422530174255`
- 判据：
  - 配置 `liftedSuccessThreshold=0.35`。
  - 三次 probe 的最大抬升均低于阈值，且没有任何 `lifted_object`。
- checkpoint 哈希：
  - `best_agent.pt`：`d140cca286eb4107348aed2b4596ee1858a88bfb7bc389ced208a1de27af8299`
  - `agent_58000.pt`：`9367999f2566fa128418b60a737c9d764b37f4363a97623d7533f6f01abb3f42`
  - `agent_60000.pt`：`1ffdbdd1bbeeb3b1e72e447964943c8e86e77cf90a1e19da9d6944025d66e4e4`
  - 本地同步文件与远端一致。
- TensorBoard event：
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/events.out.tfevents.1780129902.3090ubuntu20-04.411124.0`
- TensorBoard 标量摘要：
  - `Reward / Total reward (mean)`：first `[24, 0.19169048964977264]`，last `[60000, 6.686055660247803]`，max `8.104086875915527`
  - `Reward / Instantaneous reward (mean)`：first `[24, 0.04416413977742195]`，last `[60000, 0.3572971820831299]`，max `0.3585854470729828`
  - `Loss / Value loss`：first `[24, 2.32133412361145]`，last `[60000, 0.23892918229103088]`
  - `Loss / Policy loss`：last `[60000, -0.00208820472471416]`
  - `Learning / Learning rate`：last `[60000, 4.38957467849832e-05]`
- 远端状态：
  - `nvidia-smi`：GPU0 `283 MiB / 24576 MiB`，GPU1 `15 MiB / 24576 MiB`，利用率均 `0%`。
  - 未发现 `train_multistate.py`、`play_multistate.py`、`train_multi_bc_deter.py` 或 `vbc_teacher_probe` 后台进程。

<!-- source: session 2026-06-04 12:35:57-12:42:31 CST +0800, new teacher post-training cause probe -->

新低层 `publiccheckrollrew_42000.pt` 对应 teacher 复查原始记录：

- 远端状态命令：
  - `ssh -p 22 -i <ssh-key-path> ubuntu@<remote-host> 'date; pgrep -af "train_multistate|train.py|vbc_teacher_probe"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader'`
- 远端状态结果：
  - 时间：`Thu Jun  4 12:35:57 CST 2026`
  - 未发现真实训练/评估进程；`pgrep` 只返回当前查询命令自身。
  - GPU0：`NVIDIA GeForce RTX 3090, 543 MiB, 24576 MiB, 0 %`
  - GPU1：`NVIDIA GeForce RTX 3090, 277 MiB, 24576 MiB, 0 %`
- 新 teacher run 配置核对：
  - 路径：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/b1z1_pickmulti.yaml`
  - `low_policy_path: data/low_policy/publiccheckrollrew_42000.pt`
  - checkpoint 目录中 `agent_60000.pt` 大小 `20622806`，时间 `2026-06-04 05:24:25`；`best_agent.pt` 大小 `20622696`，时间 `2026-06-04 04:10:51`。
- 训练日志严格成功率过滤：
  - 命令核心：`awk '/^Total success rate / {v=$4+0; n++; if(n==1) first=v; last=v; if(v>max) max=v} END {...}'`
  - 结果：`strict_success_count=59917 first=0.00000000 last=0.00582955 max=0.01045189`
- 探针启动细节：
  - 直接 `conda run -n vbc-py38 python /tmp/vbc_teacher_probe.py ...` 失败：`ModuleNotFoundError: No module named 'isaacgym'`。
  - 修正方式：复用远端启动脚本环境，先 `source /home/ubuntu/vbc-remote/remote-run/remote/common.sh; activate_env`，以便设置 `PYTHONPATH`。
- 探针命令：
  - `cd /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level`
  - `CUDA_VISIBLE_DEVICES=0 python /tmp/vbc_teacher_probe.py --checkpoint /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/checkpoints/agent_60000.pt --steps 300 --experiment-dir /tmp/vbc-probe-eval --wandb-name probe-agent60000-300`
- 探针汇总：
  - `num_envs=34`
  - `steps=300`
  - `new_episodes=172`
  - `new_success=0`
  - `window_success_rate=0.0`
  - `terminated_count=172`
  - `truncated_count=0`
  - `max_curr_height=0.10838823020458221`
  - `min_curr_dist=0.07303033769130707`
  - `max_lifted_now_count=0`
  - `max_lifted_object_count=0`
  - `mean_reward_first_last=[0.21302028000354767, 0.4792332351207733]`
  - `mean_base_object_distance_first_last=[0.6222650408744812, 0.42952239513397217]`
  - `mean_command_first_last=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.007868682034313679]]`
- 动作统计摘录：
  - gripper 动作是第 7 维，即 `actions[:, 6]`。
  - step 0：gripper mean `0.402829110622406`
  - step 10：gripper mean `-0.08033773303031921`
  - step 20：gripper mean `-0.5005708336830139`
  - step 50：gripper mean `-0.3903498351573944`
  - step 100：gripper mean `-0.4061433970928192`
  - step 299：gripper mean `-0.47127673029899597`
- 代码判据：
  - `high-level/envs/b1z1_pickmulti.py` 中 `liftedSuccessThreshold=0.35` 来自配置。
  - `check_termination()` 使用 `(cube_height - table_heights - init_height) > lifted_success_threshold` 且末端距离 `< 0.1` 作为 `lifted_object`。
  - `high-level/envs/b1z1_base.py` 中 `set_gripper()` 规定 `actions[:, 6] >= 0` 打开，`< 0` 关闭；`stop_pick` 下 `actions[:, 6] < 0` 会把 base commands 置零。
- 结论：
  - 当前 checkpoint 会靠近并有闭爪尝试，但最大抬升远低于 `0.35m`，没有形成有效拾取。

<!-- source: session 2026-06-04 12:50:00-13:03:00 CST +0800, README/W&B comparison and termination reason probe -->

README 与 W&B 对比原始记录：

- README 高层 teacher 命令：
  - `python train_multistate.py --rl_device "cuda:0" --sim_device "cuda:0" --timesteps 60000 --headless --task B1Z1PickMulti --experiment_dir b1-pick-multi-teacher --wandb --wandb_project "b1-pick-multi-teacher" --wandb_name "some descriptions" --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`
- 本次远端 teacher 启动脚本：
  - `remote-run/remote/50_start_teacher_tmux.sh`
  - `remote-run/remote/common.sh::teacher_args()`
  - 与 README 核心 flags 一致；主要差异是 `VBC_USE_WANDB=0` 时不传 `--wandb --wandb_project`，避免远端网络/W&B 登录阻塞。
- W&B API 查询：
  - URL：`https://api.wandb.ai/graphql`
  - entity/project：`ericonaldo/b1-pick-multi-teacher`
  - runCount：`14`
- W&B 成功 run 摘要：
  - `publiccheckrollrew_37000_2`：`Reward / Total reward (mean)=93.53524780273438`，`Reward / Instantaneous reward (mean)=1.1190896034240725`，`Episode / Total timesteps (mean)=82.18083190917969`，成功率：Ball `0.8540629`、Bottle `0.7141111`、Bowl `0.5360954`、Cup `0.8785627`、Drill `0.5933264`、LongBox `0.6445903`、SquareBox `0.8814104`。
  - `publiccheckrollrew_37600`：`Reward / Total reward (mean)=97.98704528808594`，成功率：Ball `0.7923395`、Bottle `0.5268336`、Bowl `0.4572082`、Cup `0.7719542`、Drill `0.3914061`、LongBox `0.4331843`、SquareBox `0.7658204`。
  - `publiccheckrollrew_38000`：`Reward / Total reward (mean)=96.93528747558594`，成功率：Ball `0.6032092`、Bottle `0.3328216`、Bowl `0.3874876`、Cup `0.6691682`、Drill `0.2414035`、LongBox `0.1160058`、SquareBox `0.7349812`。
- W&B 失败/低表现 run 摘要：
  - `publiccheckrollrew_42000`：状态 `crashed`，`global_step=42504`，`Reward / Total reward (mean)=21.861331939697266`，各类成功率接近 `0`。
  - `publiccheckrollrew_39000`：`Reward / Total reward (mean)=57.57302474975586`，大多数类别成功率接近 `0`，Bowl `0.03097`、Cup `0.00988`。
- W&B 成功 run `publiccheckrollrew_37000_2` 配置/依赖：
  - `config.yaml` 显示 PPO 参数与 `train_multistate.py` 默认基本一致：`rollouts=24`、`learning_epochs=5`、`mini_batches=6`、`learning_rate=0.0005`、`checkpoint_interval=500`、`timesteps=60000`。
  - `state_preprocessor_kwargs.size` 为 `Box(-inf, inf, (1094,), float32)`，policy net 第一层为 `Linear(in_features=198, ...)`。这与 feature encoder 先把 1024 维物体 feature 编码到 128 的当前代码逻辑一致，不是 `--no_feature` run。
  - `requirements.txt` 显示：`python 3.8.18`、`torch==2.1.2`、`torchvision==0.16.2`、`torchaudio==2.1.2`、`isaacgym==1.0rc4`、`numpy==1.24.4`、`skrl==1.0.0`、`wandb==0.16.2`。
  - 本次远端高层 probe 输出显示 `torch 2.4.1+cu121`；该差异不一定是主因，但若要严格复刻 W&B 成功 run，应优先对齐。
- 本次新 teacher TensorBoard event：
  - event：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/events.out.tfevents.*`
  - `Reward / Total reward (mean)`：first `[24, 0.18909040093421936]`，last `[60000, 8.25648021697998]`，max `[50088, 9.512650489807129]`。
  - `Reward / Instantaneous reward (mean)`：first `[24, 0.05505765974521637]`，last `[60000, 0.3947032690048218]`，max `[44592, 0.40231242775917053]`。
  - `Episode / Total timesteps (mean)`：first `[24, 1.3037500381469727]`，last `[60000, 21.076250076293945]`，max `[46176, 24.690834045410156]`。
  - `Loss / Value loss`：last `[60000, 0.1458204686641693]`。
- termination probe 临时脚本：
  - 本地：`/tmp/vbc_termination_probe.py`
  - 远端：`/tmp/vbc_termination_probe.py`
  - 本地语法检查：`python3 -m py_compile /tmp/vbc_termination_probe.py` 通过。
- termination probe 命令：
  - `CUDA_VISIBLE_DEVICES=0 python /tmp/vbc_termination_probe.py --checkpoint .../teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/checkpoints/agent_60000.pt --steps 300 --experiment-dir /tmp/vbc-termination-probe --wandb-name termination-agent60000-300`
- termination probe 结果：
  - `new_episodes=172`
  - `new_success=0`
  - `window_success_rate=0.0`
  - reset cause counts：`reset=172`、`cube_falls=170`、`ik_fail=2`、`timeout=0`、`roll=0`、`pitch=0`、`base_z=0`、`lifted_now=0`、`lifted_object=0`。
  - step 20：`mean_goal_z=0.19385242462158203`，`mean_actual_ee_z_local=0.17499107122421265`，`gripper_action_mean=-0.5005708336830139`，`max_curr_height=0.07334934175014496`。
  - step 50：`mean_goal_z=0.1228995993733406`，`mean_actual_ee_z_local=0.08980337530374527`，`gripper_action_mean=-0.3903498947620392`。
  - step 299：`mean_goal_z=0.14313243329524994`，`mean_actual_ee_z_local=0.10713949799537659`，`gripper_action_mean=-0.47127678990364075`。
- 对比结论：
  - 本次 teacher reward、成功率、episode length 均显著低于 W&B 成功 run。
  - 短 episode 主要来自 `cube_falls`；策略在低位闭爪/接触后让物体掉落，而不是把物体抬高。

<!-- source: session 2026-06-04 15:13:00-15:29:19 CST +0800, validation after cube_falls fix -->

官方 `cube_falls` 修复后的验证原始记录：

- 本地代码确认：
  - 命令：`grep -n "cube_falls" Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - 输出：`403:        cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`
- 远端代码与配置确认：
  - 命令：`grep -n "cube_falls" /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - 输出：`403:        cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`
  - 命令：`grep -n "low_policy_path" /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
  - 输出：`18:  low_policy_path: data/low_policy/publiccheckrollrew_37000.pt`
- 修复后 termination probe 对旧失败 teacher 的对照结果：
  - steps：`300`
  - `new_episodes=10`
  - `new_success=0`
  - `reset=10`
  - `cube_falls=6`
  - `cube_below_table=163`
  - `ik_fail=3`
  - `timeout=0`
  - 解释：`cube_falls` 是新官方口径；`cube_below_table` 是临时 probe 中保留的旧严格口径统计。
- full-scale teacher smoke：
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log`
  - 条件：`10240` env、`24` timesteps。
  - 结果：退出码 `0`。
- 新 teacher 早期监控：
  - 日志尾部可见进度约 `191/60000`。
  - 最新 `Total success rate 0.0`。
  - GPU1 约 `9357MiB`、利用率 `92%`。
  - 尚未生成 `agent_*.pt`；代码确认 teacher `checkpoint_interval=500`。
- 2026-06-04 15:37:45 CST 早期里程碑：
  - 日志尾部可见进度约 `610/60000`。
  - 最近严格 `Total success rate` 约 `0.0001717`。
  - `agent_500.pt` 已生成，大小 `20622586` bytes。
  - 该值只说明早期不再全为 0，尚不能作为 teacher 可用证据。
- 2026-06-04 15:40:53 CST 早期状态：
  - 日志尾部可见进度约 `749/60000`。
  - 最近严格 `Total success rate` 约 `0.0002670`。
  - 仍然只属于很早期低成功率信号，尚不能判定 teacher 训练成功。

<!-- source: session 2026-06-05 12:32:27-13:53:03 CST +0800, checkpoint probes after cubefallfix teacher completion -->

修复后 teacher probe 原始记录：

- `agent_60000.pt` 300-step termination probe：
  - checkpoint：`.../teacher-cubefallfix-low37000-20260604-1530/checkpoints/agent_60000.pt`
  - `steps=300`
  - `num_envs=34`
  - `new_episodes=40`
  - `new_success=5`
  - `window_success_rate=0.125`
  - counts：`reset=40`、`cube_falls=35`、`cube_below_table=877`、`ik_fail=0`、`lifted_now=9`、`lifted_object=5`、`timeout=0`、`roll=0`、`pitch=0`、`base_z=0`
- `best_agent.pt` 直接 probe 失败：
  - 错误：`ValueError: invalid literal for int() with base 10: 'agent'`
  - 根因：`train_multistate.py` 解析 checkpoint 文件名最后一段为整数。
- `best_agent.pt` 时间戳与 `agent_53000.pt` 对齐：
  - `agent_53000.pt 20622806 2026-06-05 09:36:26.5730486660`
  - `best_agent.pt 20622696 2026-06-05 09:36:26.5930497340`
- 使用 `/tmp/best_53000.pt -> best_agent.pt` 绕过文件名 bug。
- `best_agent.pt` 300-step termination probe：
  - checkpoint：`/tmp/best_53000.pt`
  - `steps=300`
  - `num_envs=34`
  - `new_episodes=12`
  - `new_success=4`
  - `window_success_rate=0.3333333333333333`
  - counts：`reset=12`、`cube_falls=8`、`cube_below_table=659`、`ik_fail=0`、`lifted_now=5`、`lifted_object=4`、`timeout=0`
- `best_agent.pt` 1000-step termination probe：
  - checkpoint：`/tmp/best_53000.pt`
  - `steps=1000`
  - `num_envs=34`
  - `new_episodes=23`
  - `new_success=7`
  - `window_success_rate=0.30434782608695654`
  - counts：`reset=23`、`cube_falls=16`、`cube_below_table=2202`、`ik_fail=0`、`lifted_now=10`、`lifted_object=7`、`timeout=0`
- 当前评估判断：
  - `best_agent.pt` 明显优于 `agent_60000.pt`。
  - 已出现真实拾取成功，和之前 `new_success=0` 的失败 run 不同。
  - 但成功率仍不足以称为论文级完全复现。

<!-- source: session 2026-06-07 14:02-20:29 CST +0800, student eval logs parsed locally and remotely -->

Student headless 限时评估原始记录摘要：

- 运行条件：
  - 远端环境：`/home/ubuntu/vbc-remote`，conda env `vbc-py38`
  - 必需图形环境：`DISPLAY=:0`
  - GPU：`CUDA_VISIBLE_DEVICES=0`
  - 入口：`play_multi_bc_deter.py`
  - 参数要点：`--graphics_device_id 0 --headless`
  - 外层限制：`timeout --foreground -s INT 420s`
- 评估语义依据：
  - `play_multi_bc_deter.py` 调用 `get_trainer(is_eval=True)` 后执行 `trainer.eval()`。
  - `learning/dagger_trainer.py::single_agent_eval()` 中 `actions = self.agents.act(states["states"], ...)`，随后 `self.env.step(actions)`；teacher 动作 `teacher_actions` 只计算但不用于推进环境。
- `agent_60000.pt`：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/student-eval-agent60000-headless420s-20260607-1421.log`
  - 本地日志：`remote-results/student-best53000-display0-20260606-1145/logs/student-eval-agent60000-headless420s-20260607-1421.log`
  - 运行进度：`1322/14000`
  - exit code：`124`
  - `Total success rate` 记录数：`220`
  - 最新 `Total success rate`：`0.016713091922005572`
  - 最高 `Total success rate`：`0.020833333333333332`
  - 类别记录数：`220`
  - 最新类别：`Bowl=0.08695652173913043`，`Ball=0.0`，`LongBox=0.0`，`SquareBox=0.0`，`Bottle=0.0`，`Cup=0.0`，`Drill=0.0`
  - 类别峰值：`Bowl=0.1`，其它类别 `0.0`
- `best_agent.pt`：
  - 远端 symlink：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145/checkpoints/best_50000.pt -> best_agent.pt`
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/student-eval-best50000-headless420s-20260607-1502.log`
  - 本地日志：`remote-results/student-best53000-display0-20260606-1145/logs/student-eval-best50000-headless420s-20260607-1502.log`
  - 运行进度：`1300/14000`
  - exit code：`124`
  - `Total success rate` 记录数：`229`
  - 最新 `Total success rate`：`0.010554089709762533`
  - 最高 `Total success rate`：`0.019230769230769232`
  - 类别记录数：`229`
  - 最新类别：`Bowl=0.023809523809523808`，`Ball=0.0`，`LongBox=0.0`，`SquareBox=0.0`，`Bottle=0.0`，`Cup=0.0`，`Drill=0.0`
  - 类别峰值：`Bowl=0.08333333333333333`，其它类别 `0.0`
- 本地同步日志 SHA256：
  - `student-best53000-display0-20260606-1145.student.log`：`7cc443f035bde10f40c51d20e62004f4dfe5e0158ea8c9685ec68f4264c5545a`
  - `student-eval-agent60000-headless420s-20260607-1421.log`：`79fdf174d0aa3e94ea3ae662a79a992c0b4dfb83d357e7d44de1a6c8b3a583ab`
  - `student-eval-best50000-headless420s-20260607-1502.log`：`5154f45555e44a218a0009e210ac74dd6ca68a04bef59970114f5c2bfa6ed1f0`
  - `student-eval-agent60000-record150-20260607-1402.log`：`407637171f95d9e7d0aacfd7a4b0e0a8fae04cb234575af40a0608629a2e9dde`
- 远端运行状态复查：
  - 无 `play_multi_bc_deter.py`、`train_multi_bc_deter.py`、`train_multistate.py` 进程。
  - GPU0：`548 MiB / 24576 MiB`，`0%`
  - GPU1：`277 MiB / 24576 MiB`，`0%`

<!-- source: session 2026-06-08 21:18-21:55 CST +0800, official-align teacher reset/table sweep diagnostics -->

official-align teacher reset/table sweep 诊断原始摘要：

- 用户任务：开始诊断最新 official-align teacher 为何训练完成后独立 probe 仍弱。
- 远端状态：
  - 时间：`2026-06-08 21:21:58 CST`
  - GPU0：`548 MiB, 0 %`
  - GPU1：`277 MiB, 0 %`
  - 无 `train_multistate.py`、`train_multi_bc_deter.py`、`play_multistate.py` 训练/评估进程。
- reset 分布采样：
  - 临时脚本：`/tmp/vbc_reset_stats.py`
  - 初次脚本失败：先 import `torch` 再 import `isaacgym`，报 `ImportError: PyTorch was imported before isaacgym modules`。
  - 修正：脚本顶部先 import `isaacgym`，再 import `torch`。
  - 有效命令核心：`CUDA_VISIBLE_DEVICES=0 RESET_STATS_NUM_ENVS=256 RESET_STATS_CYCLES=3 python /tmp/vbc_reset_stats.py > /tmp/vbc_reset_stats.out 2> /tmp/vbc_reset_stats.err`
  - reset stats 结果摘要：
    - `table_heights` cycle0：`min=0.00198`、`mean=0.29555`、`max=0.59939`
    - `table_heights` cycle1：`min=0.00008`、`mean=0.31211`、`max=0.59837`
    - `table_heights` cycle2：`min=0.00044`、`mean=0.30334`、`max=0.59966`
    - `cube_z_minus_table_minus_init_height` 均值约 `0.00015-0.00018m`，p90 约 `0.00055m`
    - `arm_base_z` 约 `0.63983m`
  - 解释：官方对齐代码实际形成桌面高度 `[0, 0.6]`；物体 reset 基本贴桌，不支持初始悬空/穿桌假设。
- 固定桌高 sweep：
  - 临时脚本：`/tmp/vbc_table_sweep_probe.py`
  - v1 作废：脚本只调用 `env.reset()`，而 `B1Z1Base.reset()` 只 reset `reset_buf` 中已标记 env；`observed_table_heights` 显示跨高度混杂。v1 输出保留在 `/tmp/vbc_table_sweep_officialalign_best44500.out` 和 `/tmp/vbc_table_sweep_cubefallfix_best53000.out`，不能作为结论依据。
  - v2 修正：每个固定桌高前设置 `raw.reset_buf[:] = 1`，强制全体 env reset；v2 每档 `observed_table_heights` min/max 与目标高度一致。
  - official-align v2 命令核心：`CUDA_VISIBLE_DEVICES=0 python /tmp/vbc_table_sweep_probe.py --checkpoint .../teacher-officialalign-low37000-20260607-2116/checkpoints/best_44500.pt --steps 300 --heights 0.10,0.20,0.30,0.40,0.50,0.60 ... > /tmp/vbc_table_sweep_officialalign_best44500_v2.out 2> /tmp/vbc_table_sweep_officialalign_best44500_v2.err`
  - cubefallfix v2 命令核心：`CUDA_VISIBLE_DEVICES=1 python /tmp/vbc_table_sweep_probe.py --checkpoint /tmp/best_53000.pt --steps 300 --heights 0.10,0.20,0.30,0.40,0.50,0.60 ... > /tmp/vbc_table_sweep_cubefallfix_best53000_v2.out 2> /tmp/vbc_table_sweep_cubefallfix_best53000_v2.err`
- fixed table sweep v2 摘要：

| checkpoint | 0.10m | 0.20m | 0.30m | 0.40m | 0.50m | 0.60m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `officialalign best_44500.pt` | `0/7` | `0/7` | `0/1` | `0/5` | `1/6` | `1/4` |
| `cubefallfix best_53000.pt` | `1/6` | `3/7` | `5/6` | `8/8` | `6/6` | `1/1` |

- 最大抬升摘要：
  - `officialalign best_44500.pt`：`0.061/0.188/0.073/0.064/0.377/0.365m`
  - `cubefallfix best_53000.pt`：`0.385/0.389/0.414/0.393/0.396/0.365m`
- 结论：
  - 同一当前评估代码和同一低层加载路径下，上一轮候选 teacher 明显强于最新 official-align teacher。
  - 最新 official-align teacher 失败不是 checkpoint 文件名、低层加载或随机高度 probe 单点误判导致。
  - 样本数仍小，不能把固定桌高成功率当作最终评测成功率；但足够支持“不要用最新 official-align teacher 继续 student 长训”的决策。

<!-- source: session 2026-06-07 21:11-21:34 CST +0800, official-align smoke and first checkpoint validation -->

官方对齐后 teacher 启动验证原始记录摘要：

- 本地 `py_compile`：
  - 命令：`python3 -m py_compile Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - 退出码：`0`
- 远端 `py_compile`：
  - 命令：`cd /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level && python3 -m py_compile envs/b1z1_pickmulti.py`
  - 退出码：`0`
- 远端关键行：
  - `envs/b1z1_pickmulti.py:107:        cube_start_pose.p.z = table_pos[-1] + obj_height`
  - `envs/b1z1_pickmulti.py:257:            rand_heights = torch_rand_float(-0.25, 0.35, (len(env_ids), 1), device=self.device)`
  - `envs/b1z1_pickmulti.py:259:            rand_heights = torch.ones((len(env_ids), 1), device=self.device, dtype=torch.float)*self.table_heights_fix - self.table_dimz`
  - `envs/b1z1_pickmulti.py:261:        self._table_root_states[env_ids, 2] += rand_heights.squeeze(1)`
  - `envs/b1z1_pickmulti.py:265:        super()._reset_actors(env_ids)`
  - `data/cfg/b1z1_pickmulti.yaml:20:  low_policy_path: "data/low_policy/publiccheckrollrew_37000.pt"`
- Full-scale smoke：
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260607-211143.log`
  - 输出要点：`REMOTE_TEACHER_SMOKE num_envs=10240 timesteps=24`、`Low level pretrained policy loaded!`、完成 `24/24`、`Teacher smoke test finished.`
  - 退出码：`0`
- 长训启动后早期验证：
  - 2026-06-07 21:25:31 CST：`vbc_teacher_g1` tmux 存在，`python train_multistate.py ... --wandb_name teacher-officialalign-low37000-20260607-2116 ...` 进程存在；GPU1 `5779MiB/24576MiB`、利用率 `92%`。
  - 2026-06-07 21:26:05 CST：进度 `42/60000`，GPU1 `8865MiB/24576MiB`、利用率 `89%`。
  - 2026-06-07 21:28:05 CST：进度 `154/60000`，GPU1 `9301MiB/24576MiB`、利用率 `93%`。
  - 2026-06-07 21:30:05 CST：进度 `262/60000`，GPU1 `9421MiB/24576MiB`、利用率 `93%`。
  - 2026-06-07 21:34:59 CST：进度 `515/60000`，最新严格 `Total success rate=5.432420686657975e-05`，`agent_500.pt` 已生成，大小约 `20M`。
