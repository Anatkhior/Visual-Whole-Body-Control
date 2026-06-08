# 评估与验证

本次验证目标是确认仓库目录可用、工作树与 `HEAD` 一致、本地 Git 对象完整性无错误。

已执行并通过的验证：

- `tar -tzf /tmp/Visual-Whole-Body-Control-869104c.tar.gz > /tmp/Visual-Whole-Body-Control-869104c.tar.list`：退出码 `0`，说明源码包可以完整遍历。
- `git -C Visual-Whole-Body-Control read-tree HEAD`：最终退出码 `0`，说明本地对象补齐后可以重建索引。
- `git -C Visual-Whole-Body-Control status --short`：空输出。
- `git -C Visual-Whole-Body-Control fsck --no-dangling`：空输出。
- `git -C Visual-Whole-Body-Control ls-files | wc -l`：输出 `2043`。
- `git -C Visual-Whole-Body-Control log -1 --oneline --decorate`：输出 `869104c (grafted, HEAD -> main, origin/main, origin/HEAD) add ack`。

未执行的验证：

- 未安装依赖。
- 未运行训练、回放、单元测试或仿真。
- 先前未下载 README 中的外部 Google Drive 权重；2026-05-25 用户已手动下载低层权重 `model_38000.pt`，并已完成只读检查。

2026-05-25 新增只读验证：

- `ls -lh /home/hjr/projects/2-Nexus/VBC/model_38000.pt`：文件存在，大小约 `2.0M`。
- `file /home/hjr/projects/2-Nexus/VBC/model_38000.pt`：显示为 `Zip archive data`，符合 PyTorch 新格式 checkpoint 的常见表现。
- `sha256sum /home/hjr/projects/2-Nexus/VBC/model_38000.pt`：`a39f77e5c91298a6e3353664f55caf7afcc2cda4c0fd82ce6659c9fff21a6e23`。
- `torch.load(..., map_location='cpu')`：可读取为 `dict`，键包括 `infos`、`iter`、`model_state_dict`、`optimizer_state_dict`；`iter=38000`，`model_state_dict` 有 41 个键。
- `conda run -n legged-nexus-py38 python -c "import isaacgym"`：可导入 Isaac Gym。
- `conda run -n legged-nexus-py38 python -c "import torch; ..."`：PyTorch 为 `2.4.1+cu121`，但 `torch.cuda.is_available()` 为 `False`，`device_count=0`。
- `conda run -n legged-nexus-py38 python -c "import skrl"`：失败，`ModuleNotFoundError: No module named 'skrl'`。
- `conda run -n legged-nexus-py38 python -c "import legged_gym; print(legged_gym.__file__)"`：导入路径为 `/home/hjr/projects/2-Nexus/Legged-Nexus/legged_gym/__init__.py`，不是本项目。

2026-05-25 17:13:01 硬件与论文资源要求核对：

- `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader`：`NVIDIA GeForce GTX 1660 SUPER, 6144 MiB, 289 MiB, 595.71.05`。
- `lscpu`：CPU 为 `12th Gen Intel(R) Core(TM) i5-12400F`，`12` 个逻辑 CPU。
- `free -h`：内存总量 `31Gi`，可用约 `24Gi`。
- `df -h /home/hjr/projects/2-Nexus/VBC`：可用磁盘约 `383G`。
- `paper/paper.html` 附录 simulation 段：teacher 使用 `10240` 并行环境，student 使用 `240` 并行环境；作者使用 GTX4090/GTX3090，GTX4090 上 teacher 约 `36` 小时、student 约 `48` 小时。
- `high-level/train_multistate.py` 与 `high-level/train_multi_bc_deter.py`：eval/debug 时 high-level 环境数降到 `34`，因此“回放已有 checkpoint”与“从头训练到论文效果”的硬件要求不同。

2026-05-26 修复后验证：

- 沙箱外执行 CUDA 检查：`torch 2.4.1+cu121`，`torch.cuda.is_available()` 为 `True`，`device_count=1`。
- NumPy 兼容性检查：`numpy.__version__` 为 `1.23.5`，`np.float` 可用但有弃用警告；这修复了 Isaac Gym 在 NumPy `1.24.4` 下报 `AttributeError: module 'numpy' has no attribute 'float'` 的问题。
- 本项目路径导入检查：`isaacgym` 指向 `Visual-Whole-Body-Control/third_party/isaacgym/python/isaacgym/__init__.py`；`legged_gym` 指向 `Visual-Whole-Body-Control/low-level/legged_gym/__init__.py`；`rsl_rl` 指向 `Visual-Whole-Body-Control/third_party/rsl_rl/rsl_rl/__init__.py`；`skrl` 指向 `Visual-Whole-Body-Control/third_party/skrl/skrl/__init__.py`。
- 低层回放 smoke test：`play.py --exptid google_drive --task b1z1 --proj_name b1z1-low --checkpoint 38000 --observe_gait_commands --flat_terrain --sim_device cuda:0 --rl_device cuda:0 --headless` 成功输出 GPU PhysX 初始化、B1+Z1 环境张量形状、ActorCritic 结构，并显示 `Loading model from: .../model_38000.pt`；随后由 `timeout` 截停。
- 高层配置检查：`high-level/data/cfg/b1z1_pickmulti.yaml` 的 `low_policy_path` 为 `data/low_policy/model_38000.pt`，该路径存在且 `torch.load` 可读取 `iter=38000`、`model_state_dict` 41 个键。
- `pip check` 仍报告 `warp-sensor 0.2.0` 缺少若干依赖；该包不在当前 low-level/high-level smoke test 的直接路径上，暂不作为 VBC 当前阻断项。

2026-05-26 较长低层 headless 回放验证：

- 回放条件：低层 B1+Z1，checkpoint `38000`，GPU PhysX，`headless`，运行 `1000` steps，对应 `20.0` 秒仿真时间。
- 最新一次可复查结果来自 2026-05-26 10:28:28 的重新运行：进程退出码 `0`；`start_pos=[35.89782, 4.04187, 0.49611]`，`final_pos=[34.79231, 1.8524, 0.53496]`。
- 运动指标：`final_xy_displacement_m=2.45274`，`approx_xy_path_m=9.32541`；过程进度输出在 step `200/400/600/800/1000` 均显示 base position / velocity 变化。
- 稳定性指标：`root_z_min_m=0.48385`，`root_z_max_m=0.55849`，`done_count=1`；这次验证不能证明完整任务成功，但能证明低层策略不是静止或立即崩溃。
- 控制/奖励指标：`mean_abs_action=0.44452`，`max_abs_action=3.52549`，`mean_reward=0.05598`，`mean_arm_reward=0.00678`。
- 命令与速度对照：`command_mean=[0.32125, 0.0, 0.15175]`，`command_abs_mean=[0.32125, 0.0, 0.46234]`，`base_lin_vel_mean=[0.37497, -0.0775, 0.0455]`，`base_lin_vel_abs_mean=[0.37574, 0.12179, 0.0791]`。

2026-05-26 headless 视频录制修复后验证：

- 修复后命令：`play.py --exptid google_drive --task b1z1 --proj_name b1z1-low --checkpoint 38000 --observe_gait_commands --flat_terrain --sim_device cuda:0 --rl_device cuda:0 --headless --record_video`。
- 输出视频：`Visual-Whole-Body-Control/low-level/logs/videos/google_drive/google_drive-0-38000.mp4`。
- 文件验证：`ls -lh` 显示约 `654K`；`file` 显示 `ISO Media, MP4 Base Media v1`。
- 解码验证：`imageio`/ffmpeg 可读取 `250` 帧，`720x480`，`25fps`，duration `10.0` 秒，shape 为 `(480, 720, 3)`。
- 非空/变化验证：首帧标准差 `44.608`；抽样帧平均差异为 `[15.305, 13.344, 9.056, 5.046, 6.379, 6.374, 2.188, 1.539, 1.643]`；首尾帧平均差异 `14.505`；首尾变化像素比例 `0.17102`。这些指标说明视频不是空白静帧或损坏容器。
- 录制命令退出码为 `0`，输出中显示使用 GPU PhysX、加载 `model_38000.pt`，并打印 `Recording video to: .../google_drive-0-38000.mp4`。

2026-05-26 高层 teacher smoke test 验证：

- 高层导入/配置检查：沙箱外执行后输出 `torch_cuda_available True`、`torch_cuda_device_count 1`，`isaacgym` 指向项目内 `third_party/isaacgym/python/isaacgym/__init__.py`，`B1Z1PickMulti` 和 `ActorCritic` 可导入，`low_policy_path data/low_policy/model_38000.pt` 存在，checkpoint `iter=38000`，`model_state_dict_keys=41`。
- 高层标准入口短 rollout：`train_multistate.py --rl_device cuda:0 --sim_device cuda:0 --timesteps 8 --headless --task B1Z1PickMulti --experiment_dir experiments-smoke --wandb_name smoke-high-level-teacher-rollout8 --debug --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick` 退出码 `0`。
- 该短 rollout 输出显示使用 GPU PhysX、加载 `Low level pretrained policy loaded!`，并完成 tqdm 进度 `8/8`，随后 `skrl` 打印 `Closing environment` / `Environment closed`。
- 高层最小环境 smoke test：临时内联脚本把 `cfg['env']['numEnvs']` 设为 `4`，完成 `reset()` 与 `8` 次 `step()`，退出码 `0`。
- 最小环境 smoke test 关键输出：`env_num_envs 4`，`env_num_actions 9`，`env_num_obs 1094`，`obs_shape_after_reset (4, 1094)`，`low_level_policy_loaded method`，step `1/4/8` 均输出 obs shape、reward、reset_sum、robot root position 和 cube position。
- 最小环境 smoke test 显存指标：`cuda_mem_allocated_mb_after_reset 9.0`，`cuda_mem_reserved_mb_after_reset 24.0`，`cuda_mem_allocated_mb_after_steps 9.03`，`cuda_mem_reserved_mb_after_steps 24.0`。这些是 PyTorch allocator 指标，不包含 Isaac Gym 非 PyTorch 显存占用。
- 负面验证：`train_multistate.py --debug --timesteps 24 ...` 退出码 `1`，在 PPO update 的 `optimizer.step()` 处报 `torch.OutOfMemoryError`。报错显示 GPU 总容量约 `5.61 GiB`，该进程约 `5.15 GiB`，只剩约 `1.75 MiB`。该失败发生在 rollout 之后的优化器更新阶段。

2026-05-30 远端 teacher smoke test 验证：

- 远端 smoke 命令核心：`ssh -p 22 -i <ssh-key-path> ubuntu@<remote-host> "cd /home/ubuntu/vbc-remote && TEACHER_SMOKE_ENVS=10240 TEACHER_SMOKE_TIMESTEPS=24 bash remote-run/remote/40_smoke_teacher.sh"`
- smoke 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260530-161843.log`
- 关键输出：
  - `REMOTE_TEACHER_SMOKE num_envs=10240 timesteps=24`
  - `Low level pretrained policy loaded!`
  - `Total success rate 0.0`
  - 24/24 步完整跑完
  - `Teacher smoke test finished.`
- 进度观测：
  - 0/24 开始后约 79 秒完成 24/24
  - 中途对象计数持续增长，但没有 OOM 或异常退出
- 退出码：`0`
- 远端监控时 `nvidia-smi` 看到 GPU0 约 `768 MiB` 已用、GPU1 约 `15 MiB` 已用，说明 smoke 在占用单卡跑完整 rollout，而不是挂死在启动阶段。

2026-05-30 远端 teacher 长训启动验证：

- 启动命令：`bash remote-run/remote/50_start_teacher_tmux.sh`
- 启动结果：`Started teacher training in tmux session: vbc_teacher_g0`
- 长训日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`
- 这一步不代表收敛完成，只说明 teacher 训练已经进入远端后台进程。

2026-05-31 远端 teacher 完成验证：

- 远端时间：`2026-05-31 12:56:19 CST +0800`
- tmux：`vbc_teacher_g0` 不再列出，说明训练命令已退出。
- 日志进度：`60000/60000 [19:52:40<00:00, 1.19s/it]`
- 日志尾部：`Teacher training exited at 2026-05-31 12:24:41 CST +0800`
- checkpoint：
  - `agent_60000.pt` 存在，大小约 `20M`，时间 `2026-05-31 12:24`
  - `best_agent.pt` 存在，大小约 `20M`，时间 `2026-05-31 11:45`
  - 常规 `agent_*.pt` 数量为 `120`
  - checkpoint 目录总占用约 `2.4G`

2026-05-31 teacher headless eval 验证：

- `agent_60000.pt`：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-agent60000-20260531-131714.log`
  - 运行到约 `1508` eval steps 后停止
  - 日志尾部 `Total success rate 0.0`
  - 判定：效果未通过
- `best_agent.pt`：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-best-agent-20260531-132628.log`
  - 直接运行失败：`ValueError: invalid literal for int() with base 10: 'agent'`
  - 根因：`train_multistate.py` eval 路径用文件名解析 checkpoint steps，`best_agent.pt` 不符合 `agent_<step>.pt` 命名。
- `agent_58000.pt`：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-agent58000-20260531-132711.log`
  - 运行到约 `1687` eval steps，外层 `timeout 500s` 结束，退出码 `124`
  - 日志尾部 `Total success rate 0.0`
  - 判定：效果未通过
- 本地同步验证：
  - 已同步 `agent_60000.pt`、`agent_58000.pt`、`best_agent.pt`
  - 本地 `legged-nexus-py38` 可用 `torch.load(..., map_location="cpu")` 读取三者，键包括 `optimizer`、`policy`、`state_preprocessor`、`value`、`value_preprocessor`

2026-05-31 17:02:46 CST +0800 进一步 teacher 探针验证：

- 远端权重哈希：
  - `best_agent.pt`：`d140cca286eb4107348aed2b4596ee1858a88bfb7bc389ced208a1de27af8299`
  - `agent_58000.pt`：`9367999f2566fa128418b60a737c9d764b37f4363a97623d7533f6f01abb3f42`
  - `agent_60000.pt`：`1ffdbdd1bbeeb3b1e72e447964943c8e86e77cf90a1e19da9d6944025d66e4e4`
- 本地同步权重的 SHA256 与远端一致，说明关键 teacher 权重已经可靠同步回本地。
- 临时 probe：`/tmp/vbc_teacher_probe.py`，复用 `train_multistate.get_trainer(is_eval=True)` 创建 34 env eval，并记录窗口成功数、episode 数、物体最大抬升、`lifted_object`、末端-物体距离和动作统计。
- `agent_60000.pt` 原 eval 起点 probe：400 steps，`new_episodes=255`，`new_success=0`，`window_success_rate=0.0`，`max_lifted_object_count=0`，`max_curr_height=0.075884`，`min_curr_dist=0.039799`。
- `agent_60000.pt` 训练起点 `x=-2.0` 对照 probe：300 steps，`new_episodes=185`，`new_success=0`，`window_success_rate=0.0`，`max_lifted_object_count=0`，`max_curr_height=0.051838`，`min_curr_dist=0.051229`。
- 真正 `best_agent.pt` 通过 `/tmp/best_58000.pt` 临时 symlink 评估：400 steps，`new_episodes=266`，`new_success=0`，`window_success_rate=0.0`，`max_lifted_object_count=0`，`max_curr_height=0.108526`，`min_curr_dist=0.019195`。
- 成功阈值来自配置 `liftedSuccessThreshold=0.35`；上述最大抬升均低于阈值，且 `max_lifted_object_count=0`，判定没有有效拾取。
- TensorBoard event：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/events.out.tfevents.1780129902.3090ubuntu20-04.411124.0`。
- TensorBoard 摘要：`Reward / Total reward (mean)` 从 `0.191690` 到 `6.686056`，最大 `8.104087`；`Reward / Instantaneous reward (mean)` 从 `0.044164` 到 `0.357297`；`Loss / Value loss` 从 `2.321334` 降到 `0.238929`。
- 训练 reward 有改善，但 headless eval 和 probe 均未出现成功拾取；不能把 reward 上升等同于 teacher 可用。
- 远端 GPU 状态：GPU0 约 `283 MiB`、GPU1 约 `15 MiB`，利用率 `0%`；未发现 teacher/student/play/probe 后台进程。

2026-06-04 12:42:31 CST +0800 新低层重训后 teacher 定向 probe：

- 远端状态：没有 `train_multistate.py`、`train.py` 或 `vbc_teacher_probe` 后台进程；GPU0 约 `543MiB/24576MiB`、GPU1 约 `277MiB/24576MiB`，利用率均 `0%`。
- 新 teacher run 配置已核对：`teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/b1z1_pickmulti.yaml` 中 `low_policy_path: data/low_policy/publiccheckrollrew_42000.pt`，不是旧的 `model_38000.pt`。
- 严格过滤训练日志中以 `Total success rate ` 开头的行：`count=59917`、`first=0.00000000`、`last=0.00582955`、`max=0.01045189`。
- 定向 probe 命令复用 `/tmp/vbc_teacher_probe.py`，通过 `source remote-run/remote/common.sh && activate_env` 设置远端环境后运行，checkpoint 为新 teacher `agent_60000.pt`。
- `agent_60000.pt` probe 结果：300 steps、34 env、`new_episodes=172`、`new_success=0`、`window_success_rate=0.0`、`terminated_count=172`、`truncated_count=0`。
- 物体与距离指标：`max_curr_height=0.10838823020458221`、`min_curr_dist=0.07303033769130707`、`max_lifted_now_count=0`、`max_lifted_object_count=0`。
- 过程指标：平均 base-object 距离从 `0.6222650408744812` 降到 `0.42952239513397217`；平均 reward 从 `0.21302028000354767` 升到 `0.4792332351207733`。
- 动作指标：teacher 第 7 维 gripper 动作在早期均值为正，后期多次转负；按 `set_gripper()` 逻辑，`>=0` 为打开，`<0` 为关闭。因此策略有靠近并尝试闭爪的迹象，但没有达到 `liftedSuccessThreshold=0.35`。
- 判定：当前失败不是 eval 起点、checkpoint 命名或低层路径未切换导致的单点误报；更像高层策略只学到接近和部分抓取动作，没有学会稳定抬升。

2026-06-04 13:03:00 CST +0800 W&B 官方曲线与 reset 原因对比：

- README 高层 teacher 官方命令与本次远端 teacher 长训参数基本一致：`train_multistate.py --timesteps 60000 --headless --task B1Z1PickMulti --experiment_dir b1-pick-multi-teacher --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`。本次主要差异是禁用 `--wandb`，并把 config 的低层路径设为重训出的 `data/low_policy/publiccheckrollrew_42000.pt`。
- W&B 官方项目 `ericonaldo/b1-pick-multi-teacher` 公开 API 显示该项目共有 14 个 run，既有失败 run，也有成功 run；不是所有官方公开 run 都成功。
- 成功 run 示例 `publiccheckrollrew_37000_2`：`Reward / Total reward (mean)=93.535`，`Reward / Instantaneous reward (mean)=1.119`，`Episode / Total timesteps (mean)=82.181`，各类成功率约 `0.536-0.881`。
- 成功 run 示例 `publiccheckrollrew_37600`：`Reward / Total reward (mean)=97.987`，各类成功率约 `0.391-0.792`。
- 失败 run 示例 `publiccheckrollrew_42000`：状态 `crashed`，`global_step=42504`，`Reward / Total reward (mean)=21.861`，各类成功率接近 `0`。
- 本次新 teacher 的 TensorBoard：`Reward / Total reward (mean)` 末值 `8.256`、峰值 `9.513`；`Reward / Instantaneous reward (mean)` 末值 `0.395`；`Episode / Total timesteps (mean)` 末值 `21.076`。这些都显著低于 W&B 成功 run。
- reset 原因 probe：`agent_60000.pt` 300 steps、34 env 中 `new_episodes=172`、`new_success=0`，reset 原因计数 `cube_falls=170`、`ik_fail=2`、`timeout=0`、`roll=0`、`pitch=0`、`base_z=0`。
- 同一 probe 的动作/位姿过程：mean goal z 从 `0.55` 很快降到 `0.12-0.19` 附近，实际末端 local z 降到 `0.09-0.17`；gripper 动作后期为负，说明在低位闭爪/接触，导致物体掉落而不是抬升。
- 判定：与 W&B 成功 run 相比，本次训练没有进入“稳定抓取成功”阶段；最直接失败机制是策略把末端目标压得过低，闭爪接触后导致 `cube_falls` reset。

2026-06-04 15:29:19 CST +0800 官方 `cube_falls` 口径修复后的验证：

- 本地代码核对：`Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py` 中 `cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`。
- 远端代码核对：同一路径的 `cube_falls` 判定与本地一致。
- 远端配置核对：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml` 中 `low_policy_path: data/low_policy/publiccheckrollrew_37000.pt`。
- 修复后 termination probe 对照旧失败 teacher：300 steps、`new_episodes=10`、`new_success=0`、`reset=10`、官方口径 `cube_falls=6`、旧严格口径 `cube_below_table=163`、`ik_fail=3`、`timeout=0`。该结果不能证明旧 teacher 可用，但证明 reset 口径修复显著改变了 episode 终止结构。
- 修复后 full-scale teacher smoke：`10240` env、`24` timesteps、退出码 `0`，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log`。
- 新 teacher 长训启动后早期状态：进度约 `191/60000`，`Total success rate 0.0`，GPU1 正常占用约 `9357MiB`、利用率 `92%`。该阶段太早，不能据此判断最终效果。
- 2026-06-04 15:37 CST 再次复查：进度约 `610/60000`，`agent_500.pt` 已生成；最近三条严格 `Total success rate` 约 `0.0001737`、`0.0001719`、`0.0001717`。这是早期非零信号，但数量级仍极低，不能视为效果通过。

2026-06-05 13:53:03 CST +0800 修复后 teacher checkpoint 效果确认：

- 训练完成状态：`60000/60000`，远端 GPU0/GPU1 空闲；最终 `agent_60000.pt` 和 `best_agent.pt` 均存在。
- 训练日志末尾严格 `Total success rate` 约 `0.0385234`。
- `agent_60000.pt` 300-step termination probe：`new_episodes=40`、`new_success=5`、`window_success_rate=0.125`、`lifted_object=5`、`lifted_now=9`、`cube_falls=35`、`cube_below_table=877`、`ik_fail=0`。
- 直接用 `best_agent.pt` probe 仍触发文件名解析 bug：`ValueError: invalid literal for int() with base 10: 'agent'`。
- 用 `/tmp/best_53000.pt -> best_agent.pt` 绕过后，`best_agent.pt` 300-step termination probe：`new_episodes=12`、`new_success=4`、`window_success_rate=0.3333333333`、`lifted_object=4`、`cube_falls=8`、`ik_fail=0`。
- `best_agent.pt` 1000-step termination probe：`new_episodes=23`、`new_success=7`、`window_success_rate=0.3043478261`、`lifted_object=7`、`lifted_now=10`、`cube_falls=16`、`cube_below_table=2202`、`ik_fail=0`。
- 判定：修复后的 teacher 已出现实际拾取能力，`best_agent.pt` 是当前最强候选；但成功率仍低于 W&B 成功 run，不应视为论文级完全复现。

2026-06-05 23:46:10 CST +0800 W&B 成功 run 配置/代码复查：

- 用户问题：为什么这一轮 teacher 成功率仍比官方论文/W&B 成功 run 低，是否还有配置不同。
- 结构化 YAML 对比：`/tmp/wandb_97yfb8x1_b1z1_pickmulti.yaml` 与本地同步的 `remote-results/teacher-cubefallfix-low37000-20260604-1530/config/b1z1_pickmulti.yaml` 只有 `/env/low_policy_path` 不同；官方是绝对路径 `/data/mhliu/.../publiccheckrollrew_37000.pt`，本轮是相对路径 `data/low_policy/publiccheckrollrew_37000.pt`。除路径外未发现关键训练数值差异。
- 训练入口对比：`/tmp/wandb_97yfb8x1_train_multistate.py` 与当前 `Visual-Whole-Body-Control/high-level/train_multistate.py` 未显示有效 diff。
- 远端 high-level 实际依赖：Python `3.8.20`，`torch 2.4.1+cu121`，CUDA `12.1`，`numpy 1.23.5`，项目内 `isaacgym`，`skrl 1.0.0`。
- W&B 成功 run 记录的依赖依据此前下载文件为：Python `3.8.18`，`torch==2.1.2`，`torchvision==0.16.2`，`torchaudio==2.1.2`，`isaacgym==1.0rc4`，`numpy==1.24.4`，`skrl==1.0.0`。
- 判定：YAML/PPO 入口不是主要差异；剩余差异集中在环境代码、依赖数值栈和低层 checkpoint 等价性。

2026-06-06 11:52:44 CST +0800 student smoke 与启动验证：

- `best_agent.pt` 与 `agent_53000.pt` SHA256 不同：`best_agent.pt=adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c`，`agent_53000.pt=8f72889dfaf84e0c9dfd917f51bee4d257f9e9c1890f818d439208375e79fae6`。因此 student 使用 `best_53000.pt -> best_agent.pt` symlink。
- Isaac Gym camera 示例验证：`multiple_camera_envs.py` 不带 `DISPLAY` 时 core dump；`DISPLAY=:0` 时退出码 `0` 并打印 camera handle/view matrix。
- student 1-step debug smoke：`DISPLAY=:0`、GPU0、`--debug --timesteps 1`，退出码 `0`，完成 `1/1` timestep。
- student 24-step smoke：`DISPLAY=:0`、GPU0、`--timesteps 24`，退出码 `0`，完成 `24/24` timestep；日志 `/home/ubuntu/vbc-remote/remote-logs/student-smoke-display0-gpu0-24step-20260606-1142.log`。
- 完整 student 长训启动后验证：tmux `vbc_student_g0` 存在，进程 `python train_multi_bc_deter.py ... --timesteps 60000 ... --teacher_ckpt_path .../best_53000.pt` 存在；GPU0 占用约 `23240MiB/24576MiB`；日志存在且最新解析进度约 `590/60000`。

2026-06-07 20:29:53 CST +0800 student 独立 headless 限时评估：

- 评估入口：远端 `play_multi_bc_deter.py`，`DISPLAY=:0`、`CUDA_VISIBLE_DEVICES=0`、`--graphics_device_id 0`、`--headless`。评估阶段由 `DAggerTrainer.single_agent_eval()` 使用 student 动作推进环境，teacher 动作只计算用于调试/对比，不喂回环境。
- 远端状态复查：没有 `play_multi_bc_deter.py`、`train_multi_bc_deter.py` 或 `train_multistate.py` 进程；GPU0 `548MiB/24576MiB`、GPU1 `277MiB/24576MiB`，利用率均 `0%`。
- `agent_60000.pt`：
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/student-eval-agent60000-headless420s-20260607-1421.log`
  - 本地同步日志：`remote-results/student-best53000-display0-20260606-1145/logs/student-eval-agent60000-headless420s-20260607-1421.log`
  - 运行进度：约 `1322/14000`
  - 外层 `timeout` 退出码：`124`
  - `Total success rate` 最新值：`0.016713091922005572`
  - `Total success rate` 峰值：`0.020833333333333332`
  - 类别最新值：`Bowl=0.08695652173913043`，`Ball/LongBox/SquareBox/Bottle/Cup/Drill=0.0`
  - 类别峰值：`Bowl=0.1`，其它类别 `0.0`
- `best_agent.pt`：
  - 远端 symlink：`.../checkpoints/best_50000.pt -> best_agent.pt`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/student-eval-best50000-headless420s-20260607-1502.log`
  - 本地同步日志：`remote-results/student-best53000-display0-20260606-1145/logs/student-eval-best50000-headless420s-20260607-1502.log`
  - 运行进度：约 `1300/14000`
  - 外层 `timeout` 退出码：`124`
  - `Total success rate` 最新值：`0.010554089709762533`
  - `Total success rate` 峰值：`0.019230769230769232`
  - 类别最新值：`Bowl=0.023809523809523808`，`Ball/LongBox/SquareBox/Bottle/Cup/Drill=0.0`
  - 类别峰值：`Bowl=0.08333333333333333`，其它类别 `0.0`
- 结论：当前 student 有少量真实成功信号，但只集中在 `Bowl` 类，总体成功率极低。该结果不足以称为论文级复现或可展示的完整效果；`agent_60000.pt` 在本限时窗口内略强于 `best_agent.pt`。
- 本地同步验证：训练日志和三份 eval/record 日志已同步到 `remote-results/student-best53000-display0-20260606-1145/logs/`，本地 SHA256 与远端一致。

2026-06-07 21:34:59 CST +0800 官方对齐后 teacher 启动验证：

- 本地静态验证：
  - `python3 -m py_compile Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py` 退出码 `0`。
  - 本地关键片段与 `/tmp/wandb_97yfb8x1_b1z1_pickmulti.py` 同时命中：初始物体 z、桌高随机范围、固定桌高分支、桌面 root z 增量更新。
- 远端静态验证：
  - 远端 `python3 -m py_compile envs/b1z1_pickmulti.py` 退出码 `0`。
  - 远端关键行检索确认 `cube_start_pose.p.z = table_pos[-1] + obj_height`、`torch_rand_float(-0.25, 0.35, ...)`、`table_heights_fix - self.table_dimz`、`+= rand_heights.squeeze(1)` 和 `low_policy_path: "data/low_policy/publiccheckrollrew_37000.pt"`。
  - 远端低层权重：`data/low_policy/publiccheckrollrew_37000.pt`，大小约 `2.0M`，SHA256 `b61e2167ea3c370668d17bbd35b641703313a0f6bc57e1dc5117c5073af69fd6`。
- Full-scale teacher smoke test：
  - 条件：`TEACHER_SMOKE_ENVS=10240`、`TEACHER_SMOKE_TIMESTEPS=24`、`VBC_USE_WANDB=0`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260607-211143.log`
  - 结果：退出码 `0`，完成 `24/24`，加载 `Low level pretrained policy loaded!`，没有 OOM 或 traceback。
- 长训启动验证：
  - run：`teacher-officialalign-low37000-20260607-2116`
  - 2026-06-07 21:25 CST 复查：tmux `vbc_teacher_g1` 存在，进程 `python train_multistate.py ... --wandb_name teacher-officialalign-low37000-20260607-2116 ...` 存在；GPU1 约 `5779MiB/24576MiB`、利用率 `92%`。
  - 2026-06-07 21:34:59 CST 复查：进度约 `515/60000`，最新严格 `Total success rate=5.432420686657975e-05`，`agent_500.pt` 已生成，大小约 `20M`。
  - 2026-06-08 09:58 CST 复查：tmux 和 `train_multistate.py` 仍正常运行；GPU1 约 `10263MiB/24576MiB`、利用率 `91%`；进度约 `36376/60000`；最新 checkpoint `agent_36000.pt` 已生成；严格解析 `Total success rate` 最新约 `0.04365`、峰值约 `0.04607`。这是训练累计指标，不能替代完训后的独立 headless/termination probe。
  - 2026-06-08 20:45 CST 复查：训练已正常完成，日志含 `Teacher training python exit code: 0` 和 `Teacher training exited at 2026-06-08 18:38:35 CST +0800`；进度 `60000/60000`；最终 `agent_60000.pt` 已生成；严格解析 `Total success rate` 末值约 `0.03259`、峰值约 `0.04607`；`best_agent.pt` 时间戳与 `agent_44500.pt` 对齐。下一步需要独立 probe/eval，而不是只看训练累计指标。
  - 2026-06-08 21:08 CST 初步独立 termination probe：
    - `best_44500.pt -> best_agent.pt`，1000 steps、34 env：`new_success=0`、`new_episodes=10`、`window_success_rate=0.0`、`max_lifted_object_count=0`、`max_curr_height≈0.087m`、退出码 `0`。
    - `agent_60000.pt`，1000 steps、34 env：`new_success=0`、`new_episodes=8`、`window_success_rate=0.0`、`max_lifted_object_count=0`、`max_curr_height≈0.093m`、退出码 `0`。
    - `agent_36000.pt`，1000 steps、34 env：`new_success=0`、`new_episodes=20`、`window_success_rate=0.0`、`max_lifted_object_count=0`、`max_curr_height≈0.083m`、退出码 `0`。
    - 结论：本轮官方对齐 teacher 的独立 probe 暂不通过，且弱于上一轮候选 teacher `best_agent.pt` 的 1000-step probe `7/23` 成功。
  - 2026-06-08 21:46 CST reset 分布采样：
    - 临时脚本：`/tmp/vbc_reset_stats.py`，先 import `isaacgym` 再 import `torch`，避免 Isaac Gym 导入顺序错误。
    - 有效样本：`256` env、`3` 轮全量 reset。
    - 桌面高度实测：每轮约 `min≈0.000-0.002m`、`mean≈0.296-0.312m`、`max≈0.598-0.600m`，符合当前官方对齐代码 `[-0.25, 0.35]` 加到 table root z 后形成的桌面高度 `[0, 0.6]`。
    - 物体相对桌面高度：`cube_z_minus_table_minus_init_height` 均值约 `0.00015m`，p90 约 `0.00055m`，说明物体 reset 后基本贴桌，不支持“物体初始悬空/穿桌导致失败”的假设。
    - 机械臂基座 z 约 `0.6398m`，物体 z 分布约 `0.02-0.72m`，高桌样本接近机械臂可操作上界；该点会增加任务难度，但不是唯一失败解释。
  - 2026-06-08 21:55 CST 固定桌高 v2 sweep：
    - 临时脚本：`/tmp/vbc_table_sweep_probe.py`；第一版发现 `env.reset()` 只 reset 已标记 env，导致跨桌高混杂，已作废；v2 在每个高度前设置 `raw.reset_buf[:] = 1` 强制全量 reset。
    - 有效性检查：v2 每档 `observed_table_heights` 的 min/max 与固定高度一致。
    - 条件：每个 checkpoint 固定桌高 `0.10/0.20/0.30/0.40/0.50/0.60m`，每档 `300` steps、`34` env。
    - `teacher-officialalign-low37000-20260607-2116/best_44500.pt`：`0.10=0/7`、`0.20=0/7`、`0.30=0/1`、`0.40=0/5`、`0.50=1/6`、`0.60=1/4`；低中桌高基本无成功，高桌只有少量成功信号。
    - `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt` 通过 `/tmp/best_53000.pt`：`0.10=1/6`、`0.20=3/7`、`0.30=5/6`、`0.40=8/8`、`0.50=6/6`、`0.60=1/1`；样本数仍小，但相对最新 official-align 明显更强。
    - 结论：最新 official-align teacher 失败不是 `best_agent.pt` 文件名 bug、低层加载失败或随机高度 probe 单点误判；同一评估环境下上一轮候选能稳定产生抬升/成功信号。
