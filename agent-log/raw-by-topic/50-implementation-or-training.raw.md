# 实现与训练原始记录

<!-- source: session 2026-05-25, local README summaries -->

本次没有修改项目代码，没有运行训练或回放。

README 中出现的代表性命令入口包括：

- 低层训练脚本：`python train.py --headless --exptid SOME_YOUR_DESCRIPTION --proj_name b1z1-low --task b1z1 --sim_device cuda:0 --rl_device cuda:0 --observe_gait_commands`
- 低层回放脚本：`python play.py --exptid SOME_YOUR_DESCRIPTION --task b1z1 --proj_name b1z1-low --checkpoint 64000 --observe_gait_commands`
- 高层 teacher 训练脚本：`python train_multistate.py --rl_device "cuda:0" --sim_device "cuda:0" --timesteps 60000 --headless --task B1Z1PickMulti ...`
- 高层 vision student 训练脚本：`python train_multi_bc_deter.py --headless --task B1Z1PickMulti --rl_device "cuda:0" --sim_device "cuda:0" --timesteps 60000 ...`

<!-- source: session 2026-05-25, simulation reproduction conclusion -->

关于“当前项目能否在这台机器上复现”的结论：

- 当前机器不能直接完整复现论文级结果。
- 如果先修环境和补 checkpoint，可以做代码级/小规模仿真复现。
- 完整训练 teacher/student 到 README 预期效果对 GTX 1660 SUPER 6GB 不现实；论文附录提到 teacher/student 在 GTX 4090/3090 级别 GPU 上训练，且 student 因视觉渲染占用大，只用 240 个并行环境。
- 真实机器人部署还需要 B1/Z1、双 RealSense、TrackingSAM 和 server-client 部署栈；本仓库当前已读路径主要覆盖训练和仿真。

建议复现阶段：

1. 环境准备：Python 3.8、Isaac Gym、PyTorch CUDA、`rsl_rl`、`skrl`、`low-level`。
2. 资产与 checkpoint 准备：低层 checkpoint、高层对象资产与 `features.npy`、必要时 teacher/student checkpoint。
3. 低层仿真回放：先验证 `model_38000.pt` 能否驱动 low-level。
4. 高层 teacher 仿真：配置 `low_policy_path` 后训练或回放 teacher。
5. 视觉 student 仿真：需要 teacher checkpoint，使用 DAgger 蒸馏。
6. 评测：使用 `play_multistate.py`、`play_multi_bc_deter.py`。

用户新增低层权重状态：

- 路径：`/home/hjr/projects/2-Nexus/VBC/model_38000.pt`
- PyTorch 可读取；包含 `iter=38000` 和 `model_state_dict`。
- 该权重可用于跳过低层从零训练，但仍不足以直接复现完整视觉拾取效果；还需要 high-level teacher/student checkpoint 或自行训练高层。

<!-- source: user question 2026-05-25 17:02:45 CST +0800, simulation-first feasibility -->

用户问题：

“我刚才已经通过google Drive把低层控制的权重文件下载下来了，路径就在VBC/ 下，名字是model_38000.pt的一个压缩包；现在请告诉我如果我现在只想首先在仿真环境里实现这个项目的效果，是否可行？”

同步结论：

- 若“项目效果”先限定为低层仿真回放：有条件可行。`model_38000.pt` 已存在且可由 PyTorch 读取，能绕过 low-level 从零训练。
- 若“项目效果”指论文展示的完整视觉拾取：当前不够。还缺 high-level teacher/student checkpoint，或需要先训练 teacher，再训练 vision student。
- 直接阻断项仍是环境，不是 checkpoint：当前 `legged-nexus-py38` 中 `torch.cuda.is_available()` 为 `False`，`skrl` 缺失，`rsl_rl` 与 `legged_gym` 指向 `/home/hjr/projects/2-Nexus/Legged-Nexus` 而非本项目。
- 建议第一阶段只做 low-level `play.py` 小规模仿真，确认 `model_38000.pt` 能驱动低层控制；第二阶段再修改 high-level 配置中的 `low_policy_path` 并进入 teacher/student。

<!-- source: user clarification 2026-05-25 17:13:01 CST +0800, full simulation effect feasibility -->

用户澄清：

“我其实想问的是这台机器的配置，能不能支持到一直到最终的仿真效果，也就是高层控制和低层控制都可以起效，从而在仿真里达到论文中的效果”

同步结论：

- 本机配置：GTX 1660 SUPER 6GB、i5-12400F、31GiB RAM、当前工作区磁盘可用约 `383G`。
- 如果已有完整 checkpoint：可以尝试小规模仿真回放，使低层和高层策略联动起效；high-level play/eval 代码会把环境数设为 `34`。
- 如果要从头训练到论文效果：不建议在本机上做。论文附录说明 teacher 使用 `10240` 并行环境，student 使用 `240` 并行视觉环境，并在 GTX4090/GTX3090 级别 GPU 上训练；GTX4090 上 teacher 约 `36` 小时、student 约 `48` 小时。
- 现实可行目标：把本机作为开发与验证机，完成依赖修复、低层回放、已训练 high-level checkpoint 回放、少量环境 smoke test。完整 teacher/student 训练应换到更强 GPU，优先 RTX 3090/4090 或更高显存机器。

<!-- source: session 2026-05-26 09:09:37 CST +0800, low-level playback repair -->

本轮实际修复：

- 为低层 `play.py` 的原有加载逻辑创建 checkpoint 链接：
  - `Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt -> /home/hjr/projects/2-Nexus/VBC/model_38000.pt`
  - 对应运行参数：`--proj_name b1z1-low --exptid google_drive --checkpoint 38000`
- 为 high-level 创建低层策略链接：
  - `Visual-Whole-Body-Control/high-level/data/low_policy/model_38000.pt -> /home/hjr/projects/2-Nexus/VBC/model_38000.pt`
- 修改 high-level 配置：
  - 文件：`Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
  - 旧值：`/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt`
  - 新值：`data/low_policy/model_38000.pt`
- 低层回放验证命令核心参数：
  - `play.py --exptid google_drive --task b1z1 --proj_name b1z1-low --checkpoint 38000 --observe_gait_commands --flat_terrain --sim_device cuda:0 --rl_device cuda:0 --headless`
- 验证结果：
  - 低层 `play.py` 可创建 GPU PhysX 仿真。
  - 可初始化 B1+Z1 环境张量。
  - 可构建 ActorCritic 网络。
  - 可加载 `/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt`。
  - 进程由 `timeout` 主动截停，未做长时间行为评估。
- high-level 验证：
  - 按真实顺序先导入 `isaacgym`，再导入 `skrl`、`B1Z1PickMulti`、`ActorCritic`，可导入成功。
  - 从 `data/cfg/b1z1_pickmulti.yaml` 读取到 `low_policy_path data/low_policy/model_38000.pt`。
  - `torch.load` 该路径可读取 checkpoint，`iter=38000`，`model_state_dict` 有 41 个键。

<!-- source: session 2026-05-26 11:44:45 CST +0800, high-level teacher smoke progression -->

高层 teacher smoke test 推进结论：

- 目标不是训练收敛，而是确认高层 state-based teacher 的关键运行链路：导入、读取配置、加载低层策略、创建 `B1Z1PickMulti`、reset、step、标准入口短 rollout。
- 标准 `train_multistate.py --debug` 会把环境数降到 `34`。本机 GTX 1660 SUPER 6GB 能创建该环境并完成短 rollout，但在 `24` timestep 后进入 PPO update 时显存不足。
- 为了确认小显存机器上仍能做更小规模环境级 smoke test，临时脚本将 `cfg['env']['numEnvs']` 改为 `4`，直接创建 `B1Z1PickMulti` 并执行 reset/step。
- 该临时验证暴露 `B1Z1PickMulti._reset_envs()` 中分类成功率统计默认 `num_envs >= 33` 的假设。修复后小规模 reset/step 通过。
- 当前代码改动仍不等同于完整 teacher 训练能力；它只是让高层环境在小规模 smoke test 下更健壮，并保留标准 debug 入口可做短 rollout 的能力。

<!-- source: session 2026-05-29 16:16:54 CST +0800, remote dual-3090 training plan -->

远端双 RTX 3090 训练计划：

- 判断：
  - 单张 RTX 3090 级别 GPU 符合论文中使用的训练硬件类别。
  - 双 RTX 3090 足够作为训练主机，但本项目没有默认多卡训练；两张卡不会自动合并为 `48GB` 显存。
- 默认策略：
  - GPU0：teacher 训练。
  - GPU1：保留给评估、第二 teacher seed，或 teacher checkpoint 可用后的 student。
- teacher 默认参数：
  - `TEACHER_TIMESTEPS=60000`
  - `TEACHER_SMOKE_ENVS=10240`
  - `TEACHER_SMOKE_TIMESTEPS=24`
  - `TEACHER_GPU=0`
  - `VBC_USE_WANDB=0`
- 远端最短启动顺序：
  1. 本机填写 `remote-run/local/remote.env`。
  2. 本机运行 `bash remote-run/local/sync_to_remote.sh`。
  3. 远端运行 `bash remote-run/remote/00_probe_remote.sh`。
  4. 远端运行 `bash remote-run/remote/10_install_env.sh`。
  5. 远端运行 `bash remote-run/remote/20_prepare_project.sh`。
  6. 远端运行 `bash remote-run/remote/30_check_imports.sh`。
  7. 远端运行 `bash remote-run/remote/40_smoke_teacher.sh`。
  8. 远端运行 `bash remote-run/remote/50_start_teacher_tmux.sh`。
- student：
  - 必须等待 teacher checkpoint。
  - 设置 `STUDENT_TEACHER_CKPT=/path/to/teacher/checkpoint.pt` 后再运行 `remote-run/remote/70_start_student_tmux.sh`。

<!-- source: session 2026-05-30 16:18:43 CST +0800, remote teacher smoke and launch -->

远端 teacher smoke 与长训启动原始记录：

- smoke 命令核心：
  - `ssh -p 22 -i <ssh-key-path> ubuntu@<remote-host> "cd /home/ubuntu/vbc-remote && TEACHER_SMOKE_ENVS=10240 TEACHER_SMOKE_TIMESTEPS=24 bash remote-run/remote/40_smoke_teacher.sh"`
- smoke 日志文件：
  - `/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260530-161843.log`
- 关键输出片段：
  - `REMOTE_TEACHER_SMOKE num_envs=10240 timesteps=24`
  - `Low level pretrained policy loaded!`
  - 多次 `success_rate` 打印均为 `0.0`
  - `Bowl count`、`Ball count` 等对象计数持续变化，说明 rollout 正常推进
  - `24/24` 完成
  - `Teacher smoke test finished.`
- 运行耗时：
  - 从开始到结束约 `79` 秒
- 长训启动：
  - `bash remote-run/remote/50_start_teacher_tmux.sh`
  - 输出 `Started teacher training in tmux session: vbc_teacher_g0`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`
- 解释：
  - smoke 的 10k 环境前段短暂沉默是正常初始化，不应误判为卡死。

<!-- source: session 2026-05-31 12:56:19 CST +0800, remote teacher training completed -->

远端 teacher 训练完成原始记录：

- 远端日志：
  - `/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`
- 日志最终进度：
  - `60000/60000 [19:52:40<00:00, 1.19s/it]`
- 退出时间：
  - `Teacher training exited at 2026-05-31 12:24:41 CST +0800`
- checkpoint 目录：
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints`
- 最终权重：
  - `agent_60000.pt`
  - `best_agent.pt`
- 数量与空间：
  - 常规 `agent_*.pt` 共 `120` 个
  - 目录总占用约 `2.4G`

<!-- source: session 2026-05-31 13:17:14-16:26:39 CST +0800, teacher eval and local sync -->

teacher 效果确认与后续动作：

- `agent_60000.pt` 评估：
  - 命令入口：`play_multistate.py --checkpoint .../agent_60000.pt --headless --task B1Z1PickMulti --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-agent60000-20260531-131714.log`
  - 约 `1500` eval steps 后 `Total success rate 0.0`
- `best_agent.pt` 评估：
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-best-agent-20260531-132628.log`
  - 失败：`ValueError: invalid literal for int() with base 10: 'agent'`
  - 原因：eval 代码从 checkpoint 文件名解析步数，不支持 `best_agent.pt` 这种文件名。
- `agent_58000.pt` 评估：
  - 用作 `best_agent.pt` 的近似候选，因为两者远端时间戳同为 `2026-05-31 11:45`。
  - 远端日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-eval-agent58000-20260531-132711.log`
  - 约 `1687` eval steps 后 `Total success rate 0.0`
- 结论：
  - teacher 效果未确认。
  - student 训练未启动，避免基于失败 teacher 浪费远端 GPU 时间。
- 本地同步：
  - `remote-results/teacher-20260530-162617/checkpoints/agent_60000.pt`
  - `remote-results/teacher-20260530-162617/checkpoints/agent_58000.pt`
  - `remote-results/teacher-20260530-162617/checkpoints/best_agent.pt`

<!-- source: session 2026-05-31 17:02:46 CST +0800, teacher probe diagnosis and student hold decision -->

追加 teacher 探针与 student 决策原始记录：

- 用户要求：继续推进，但效果确认后先把权重同步回本地，再继续远端 student 训练。
- 执行策略：先做 teacher 效果确认；由于 teacher 未通过，不启动 student。
- 真正 best 权重评估：
  - 远端临时链接：`/tmp/best_58000.pt -> /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/best_agent.pt`
  - 原因：`best_agent.pt` 文件名会触发 `train_multistate.py` 的 checkpoint step 解析错误。
  - 结果：400 steps、34 env、266 个新 episode，`new_success=0`，`window_success_rate=0.0`，`max_lifted_object_count=0`，最大抬升约 `0.109m`。
- `agent_60000.pt` probe：
  - 原 eval 起点：400 steps、255 个新 episode，`new_success=0`，最大抬升约 `0.076m`。
  - 训练起点 `x=-2.0` 对照：300 steps、185 个新 episode，`new_success=0`，最大抬升约 `0.052m`。
- 权重同步状态：
  - 本地已同步 `best_agent.pt`、`agent_58000.pt`、`agent_60000.pt`。
  - 本地 SHA256 与远端一致。
- 决策：
  - 当前 teacher 未达到作为专家的最低条件。
  - 不启动 `train_multi_bc_deter.py` student 训练，避免把失败 teacher 蒸馏给 student。
  - 三个文件均可被本地 `legged-nexus-py38` 环境 `torch.load(..., map_location="cpu")` 读取。

<!-- source: session 2026-05-31 18:41:27-18:58:00 CST +0800, remote status and teacher metric recheck -->

远端状态与 teacher 训练指标复查原始记录：

- 本地时间：`2026-05-31 18:41:27 +0800`。
- 远端时间：`2026-05-31 18:43:44 +0800`。
- 远端 GPU：
  - GPU0 `NVIDIA GeForce RTX 3090`，约 `285 MiB / 24576 MiB`，利用率约 `2%`。
  - GPU1 `NVIDIA GeForce RTX 3090`，约 `15 MiB / 24576 MiB`，利用率约 `0%`。
- 远端活动进程检查：
  - 命令匹配 `train_multistate.py|play_multistate.py|train_multi_bc_deter.py|play_multi_bc_deter.py|evaluate_sea_nav_success.py`。
  - 结果：`no_active_process`。
- 远端 checkpoint：
  - 目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints`
  - `du -sh`: `2.4G`
  - `agent_*.pt`: `120`
  - `best_agent.pt`、`agent_58000.pt`、`agent_60000.pt` 均约 `20M`
- 远端关键权重 SHA256：
  - `best_agent.pt`: `d140cca286eb4107348aed2b4596ee1858a88bfb7bc389ced208a1de27af8299`
  - `agent_58000.pt`: `9367999f2566fa128418b60a737c9d764b37f4363a97623d7533f6f01abb3f42`
  - `agent_60000.pt`: `1ffdbdd1bbeeb3b1e72e447964943c8e86e77cf90a1e19da9d6944025d66e4e4`
- 本地同步校验：
  - `remote-results/teacher-20260530-162617/checkpoints/best_agent.pt`
  - `remote-results/teacher-20260530-162617/checkpoints/agent_58000.pt`
  - `remote-results/teacher-20260530-162617/checkpoints/agent_60000.pt`
  - 三者 SHA256 与远端一致。
- TensorBoard event：
  - 路径：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/events.out.tfevents.1780129902.3090ubuntu20-04.411124.0`
  - `Reward / Instantaneous reward (mean)`: 首值 `(24, 0.0441641)`，末值 `(60000, 0.357297)`，峰值 `(59616, 0.358585)`。
  - `Reward / Total reward (mean)`: 首值 `(24, 0.19169)`，末值 `(60000, 6.68606)`，峰值 `(58656, 8.10409)`。
  - `Loss / Value loss`: 首值 `(24, 2.32133)`，末值 `(60000, 0.238929)`。
- 完整训练日志成功率解析：
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`
  - 可解析 `Total success rate` 条目：`59949`
  - 首值：`0.0`
  - 末值：`0.002522652561991047`
  - 峰值：`0.004016529528294751`
- 训练 run 保存配置：
  - `numEnvs: 10240`
  - `maxEpisodeLength: 150`
  - `controlFrequencyLow: 8`
  - `liftedSuccessThreshold: 0.35`
  - `low_policy_path: "data/low_policy/model_38000.pt"`
  - reward scales: `pick_up: 3.5`、`command_penalty: -1.0`、`command_reward: 0.25`、`standpick: 0.25`

<!-- source: session 2026-05-31 23:57:10 CST +0800, low-level retraining start -->

低层重训原始执行要点：

- 用户要求：按作者低层环境重新训练一个匹配的低层策略，之后再重训 teacher。
- 远端现有 `vbc-py38` 检查：Python `3.8.20`、`torch 2.4.1+cu121`、`numpy 1.23.5`、`wandb 0.24.2`，与 W&B 公开 run 的低层环境不一致。
- 新增并同步到远端的脚本：
  - `remote-run/remote/90_install_low_level_env.sh`
  - `remote-run/remote/91_smoke_low_level.sh`
  - `remote-run/remote/92_start_low_level_tmux.sh`
  - `remote-run/remote/93_monitor_low_level.sh`
  - `remote-run/remote/94_prepare_teacher_from_low.sh`
  - `remote-run/remote/95_wait_low_then_start_teacher.sh`
  - `remote-run/remote/96_start_wait_low_then_teacher_tmux.sh`
- 新环境安装命令：`LOW_CONDA_ENV_NAME=vbc-low-cu113 bash remote-run/remote/90_install_low_level_env.sh`。
- 新环境验证输出：`torch 1.10.0+cu113 cuda 11.3 cuda_available True`，`numpy 1.23.5`，`wandb 0.16.2`。
- 低层 smoke 命令：`LOW_CONDA_ENV_NAME=vbc-low-cu113 LOW_GPU=0 LOW_SMOKE_ITERATIONS=2 bash remote-run/remote/91_smoke_low_level.sh`。
- 低层 smoke 结果：`Learning iteration 0/2` 和 `Learning iteration 1/2` 完成，`Total timesteps` 到 `6144`，退出码 `0`。
- 首次低层长训 run `publiccheckrollrew-retrain-20260531-2350` 因 W&B online init 超时退出。
- 修复后低层长训命令：`LOW_RUN_NAME=publiccheckrollrew-retrain-20260531-2354 LOW_CONDA_ENV_NAME=vbc-low-cu113 LOW_GPU=0 LOW_NUM_ENVS=6144 LOW_MAX_ITERATIONS=45000 LOW_WANDB_MODE=disabled bash remote-run/remote/92_start_low_level_tmux.sh`。
- 当前低层长训监控：
  - tmux：`vbc_low_g0`
  - run：`publiccheckrollrew-retrain-20260531-2354`
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354`
  - log：`/home/ubuntu/vbc-remote/remote-logs/publiccheckrollrew-retrain-20260531-2354.low.log`
  - 2026-05-31 23:56 CST：`Learning iteration 33/45000`，GPU0 `9358MiB/24576MiB`、`84%`。
  - 已生成：`model_0.pt 679978 bytes`。
- 自动 teacher watcher 命令：`LOW_RUN_NAME=publiccheckrollrew-retrain-20260531-2354 LOW_CHECKPOINT_STEP=42000 TEACHER_GPU=1 TEACHER_RUN_NAME=teacher-after-publiccheckrollrew-retrain-20260531-2354-42000 bash remote-run/remote/96_start_wait_low_then_teacher_tmux.sh`。
- watcher：
  - tmux：`vbc_wait_low_teacher`
  - log：`/home/ubuntu/vbc-remote/remote-logs/wait-publiccheckrollrew-retrain-20260531-2354-then-teacher.log`
  - 等待：`model_42000.pt`
  - 动作：复制为高层 `data/low_policy/publiccheckrollrew_42000.pt`，更新远端高层 cfg，然后在 GPU1 启动 teacher run `teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`。

<!-- source: session 2026-06-02 20:56:59-21:03:36 CST +0800, low-level retraining second resume -->

低层重训第二次恢复原始记录：

- 用户追问：`现在训练的进度如何？`
- 远端时间：`2026-06-02 20:56:59 CST +0800`。
- 低层监控输出：
  - tmux `vbc_low_g0`: `not running`
  - GPU0：约 `543MiB/24576MiB`，利用率 `0%`
  - GPU1：约 `277MiB/24576MiB`，利用率 `0%`
  - 最新 checkpoint 列表末尾：`model_34400.pt`，约 `2032203 bytes`。
  - 日志末尾训练进度：`Learning iteration 34529/45000`，`Mean reward: 18.53`，`Total timesteps: 1641185280`。
- watcher 日志末尾：
  - `2026-06-02 17:54:06 CST +0800 low-level tmux session is not running and checkpoint is missing: vbc_low_g0`
  - 说明 `model_42000.pt` 未生成，teacher 未启动。
- teacher 监控输出：
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`
  - `No checkpoint directory yet`
  - `No log file yet`
- `pgrep -af train.py` 返回空，确认没有残留训练进程。
- 远端 run dir 列表确认最新可用低层 checkpoint 为 `model_34400.pt`，目录总占用约 `335M`。
- 恢复命令参数：
  - `LOW_RUN_NAME=publiccheckrollrew-retrain-20260531-2354`
  - `LOW_RESUME_RUN_NAME=publiccheckrollrew-retrain-20260531-2354`
  - `LOW_RESUME_CHECKPOINT=34400`
  - `LOW_MAX_ITERATIONS=10600`
  - `LOW_NUM_ENVS=6144`
  - `LOW_GPU=0`
  - `LOW_WANDB_MODE=disabled`
  - 脚本：`/home/ubuntu/vbc-remote/remote-run/remote/92_start_low_level_tmux.sh`
- 恢复启动输出：
  - `Started low-level training in tmux session: vbc_low_g0`
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354`
  - log：`/home/ubuntu/vbc-remote/remote-logs/publiccheckrollrew-retrain-20260531-2354.low.log`
- watcher 重启参数：
  - `LOW_CHECKPOINT_STEP=42000`
  - `TEACHER_GPU=1`
  - `TEACHER_RUN_NAME=teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`
  - 脚本：`/home/ubuntu/vbc-remote/remote-run/remote/96_start_wait_low_then_teacher_tmux.sh`
- watcher 重启输出：
  - `Started low-to-teacher watcher in tmux session: vbc_wait_low_teacher`
  - 等待：`model_42000.pt`
  - log：`/home/ubuntu/vbc-remote/remote-logs/wait-publiccheckrollrew-retrain-20260531-2354-then-teacher.log`
- 恢复后确认：
  - tmux：`vbc_low_g0`、`vbc_wait_low_teacher` 均存在。
  - 低层监控时间：`2026-06-02 21:03:36 CST +0800`。
  - GPU0：约 `9610MiB/24576MiB`，利用率 `93%`。
  - GPU1：约 `277MiB/24576MiB`，利用率 `0%`。
  - 当前日志：`Learning iteration 34415/45000`，`Mean reward: 14.53`。
  - watcher 日志：`2026-06-02 21:02:38 CST +0800 checkpoint not ready; sleeping 300s`。

<!-- source: session 2026-06-03 08:42:31-08:44:59 CST +0800, low-level completion and teacher restart -->

低层完成与新 teacher 启动原始记录：

- 用户追问：`现在进展如何？`
- 远端低层监控时间：`2026-06-03 08:42:31 CST +0800`。
- 远端 tmux：查询时 `no server running on /tmp/tmux-1000/default`，说明低层和 watcher 均已退出。
- 低层 GPU 状态：
  - GPU0：约 `542MiB/24576MiB`，利用率 `0%`。
  - GPU1：约 `277MiB/24576MiB`，利用率 `0%`。
- 低层最新 checkpoint：
  - `model_42800.pt`，`2026-06-03 03:26:29 CST`
  - `model_43000.pt`，`2026-06-03 03:35:41 CST`
  - `model_43200.pt`，`2026-06-03 03:44:54 CST`
  - `model_43400.pt`，`2026-06-03 03:54:07 CST`
  - `model_43600.pt`，`2026-06-03 04:03:18 CST`
  - `model_43800.pt`，`2026-06-03 04:12:29 CST`
  - `model_44000.pt`，`2026-06-03 04:21:40 CST`
  - `model_44200.pt`，`2026-06-03 04:30:51 CST`
  - `model_44400.pt`，`2026-06-03 04:40:00 CST`
  - `model_44600.pt`，`2026-06-03 04:49:15 CST`
  - `model_44800.pt`，`2026-06-03 04:58:29 CST`
  - `model_45000.pt`，`2026-06-03 05:07:40 CST`
- 低层日志末尾：
  - `Learning iteration 44999/45000`
  - `Mean reward: 18.44`
  - `Total timesteps: 1563033600`
  - `Low-level training exited at 2026-06-03 05:07:47 CST +0800`
- watcher 日志：
  - `2026-06-03 02:52:38 CST +0800 found checkpoint: .../model_42000.pt`
  - 随后失败：`/home/ubuntu/vbc-remote/remote-run/remote/94_prepare_teacher_from_low.sh: 行 23: python：未找到命令`
- teacher 监控在修复前显示：
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`
  - `No checkpoint directory yet`
  - `No log file yet`
- 本地修复：
  - 修改 `remote-run/remote/94_prepare_teacher_from_low.sh`，在 Python YAML 更新前调用 `activate_env`。
  - 已用 `scp` 同步到远端同路径。
- 手动补跑低层权重准备：
  - 命令参数：`LOW_RUN_NAME=publiccheckrollrew-retrain-20260531-2354 LOW_CHECKPOINT_STEP=42000 CONDA_ENV_NAME=vbc-py38`
  - 输出 source：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_42000.pt`
  - 输出 target：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_42000.pt`
  - 输出 cfg：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
- 新 teacher 启动：
  - 命令参数：`TEACHER_GPU=1 TEACHER_RUN_NAME=teacher-after-publiccheckrollrew-retrain-20260531-2354-42000 TEACHER_TIMESTEPS=60000 VBC_USE_WANDB=0 CONDA_ENV_NAME=vbc-py38`
  - 输出：`Started teacher training in tmux session: vbc_teacher_g1`
  - log：`/home/ubuntu/vbc-remote/remote-logs/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000.teacher.log`
- teacher 启动后确认：
  - 远端时间：`2026-06-03 08:44:59 CST +0800`
  - tmux：`vbc_teacher_g1`
  - GPU1：约 `8733MiB/24576MiB`，利用率 `94%`
  - 日志进度：约 `41/60000`
  - 早期 `Total success rate 0.0`
  - checkpoint 目录当时尚未生成。
- 2026-06-03 08:47:28 CST 二次确认：
  - tmux `vbc_teacher_g1` 仍在运行。
  - GPU1：约 `9219MiB/24576MiB`，利用率 `94%`。
  - 日志进度：约 `177/60000`。
  - `Total success rate 0.0`。
- 2026-06-03 12:31:05 CST 再次确认：
  - tmux `vbc_teacher_g1` 仍在运行。
  - GPU1：约 `10193MiB/24576MiB`，利用率 `91%`。
  - checkpoint 列表末尾：`agent_7500.pt`、`agent_8000.pt`、`agent_8500.pt`、`agent_9000.pt`、`agent_9500.pt`、`agent_10000.pt`、`agent_10500.pt`、`agent_11000.pt`、`agent_11500.pt`，另有 `best_agent.pt`。
  - 日志进度：约 `11507/60000`。
  - 最新可见 `Total success rate 0.008456691379717617`。
- 2026-06-03 18:59:56 CST 再次确认：
  - tmux `vbc_teacher_g1` 仍在运行。
  - GPU1：约 `10193MiB/24576MiB`，利用率 `96%`。
  - checkpoint 列表末尾：`agent_26500.pt`、`agent_27000.pt`、`agent_27500.pt`、`agent_28000.pt`、`agent_28500.pt`、`agent_29000.pt`、`agent_29500.pt`、`agent_30000.pt`、`agent_30500.pt`，另有 `best_agent.pt`。
  - 日志进度：约 `30631/60000`。
  - 最新可见 `Total success rate 0.010280826476492649`。
- 2026-06-04 09:12:43 CST 完成状态确认：
  - `tmux ls` 输出：`no server running on /tmp/tmux-1000/default`。
  - GPU0：约 `541MiB/24576MiB`，利用率 `0%`。
  - GPU1：约 `277MiB/24576MiB`，利用率 `0%`。
  - checkpoint 列表末尾：`agent_56000.pt`、`agent_56500.pt`、`agent_57000.pt`、`agent_57500.pt`、`agent_58000.pt`、`agent_58500.pt`、`agent_59000.pt`、`agent_59500.pt`、`agent_60000.pt`，另有 `best_agent.pt`。
  - `agent_60000.pt` 时间：`2026-06-04 05:24:25 CST`。
  - `best_agent.pt` 时间：`2026-06-04 04:10:51 CST`。
  - 日志末尾：`Teacher training exited at 2026-06-04 05:24:26 CST +0800`。
  - 末尾可见 `Total success rate 0.005829552175302348`。

<!-- source: session 2026-06-04 15:13:00-15:29:19 CST +0800, cube_falls fix and new teacher start -->

官方 `cube_falls` 修复与新 teacher 启动原始记录：

- W&B 成功 run `publiccheckrollrew_37000_2` 保存代码中的关键差异：
  - 官方：`cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`
  - 当前失败训练前：`cube_falls = z_cube < self.table_heights`
- 本地修复文件：
  - `Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - `grep -n "cube_falls"` 输出第 `403` 行为官方口径。
- 远端修复确认：
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - `grep -n "cube_falls"` 输出第 `403` 行为官方口径。
- 远端高层配置确认：
  - 文件：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
  - `low_policy_path: data/low_policy/publiccheckrollrew_37000.pt`
- 修复后 full-scale smoke：
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log`
  - 条件：`10240` env，`24` timesteps。
  - 结果：退出码 `0`。
- 新 teacher 长训：
  - tmux：`vbc_teacher_g1`
  - run：`teacher-cubefallfix-low37000-20260604-1530`
  - GPU：`CUDA_VISIBLE_DEVICES=1`
  - 命令核心：`python train_multistate.py --rl_device cuda:0 --sim_device cuda:0 --timesteps 60000 --headless --task B1Z1PickMulti --experiment_dir b1-pick-multi-teacher --wandb_name teacher-cubefallfix-low37000-20260604-1530 --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-cubefallfix-low37000-20260604-1530.teacher.log`
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530`
- 2026-06-04 15:29:19 CST 监控：
  - tmux `vbc_teacher_g1` 存在。
  - 训练进程：`python train_multistate.py ...`。
  - GPU0：约 `543MiB`，利用率 `0%`。
  - GPU1：约 `9357MiB`，利用率 `92%`。
  - 日志进度末尾：`187/60000` 到 `191/60000`。
  - 最新严格 `Total success rate`：`0.0`。
  - checkpoint 查询无 `agent_*.pt` 输出，符合尚未到 `checkpoint_interval=500` 的状态。
- 2026-06-04 15:37:45 CST 监控：
  - 日志进度末尾：`608/60000`、`609/60000`、`610/60000`。
  - 最新严格 `Total success rate`：`0.00017366806244331666`、`0.0001718967855301106`、`0.00017170984851375587`。
  - checkpoint：`agent_500.pt 20622586 2026-06-04 15:35:41.0808106280`。
- 2026-06-04 15:40:53 CST 监控：
  - 日志进度末尾：`747/60000`、`748/60000`、`749/60000`。
  - 最新严格 `Total success rate`：`0.0002675943270002676`、`0.0002672619796252044`、`0.0002670017276582378`。
  - GPU0：约 `540MiB`，利用率 `0%`。
  - GPU1：约 `10209MiB`，利用率 `93%`。

<!-- source: session 2026-06-05 12:32:27-13:53:03 CST +0800, completed cubefallfix teacher and local sync -->

修复后 teacher 完成与同步原始记录：

- 远端时间：`2026-06-05 12:32:27 CST +0800`。
- 远端 GPU：GPU0 约 `543MiB`、利用率 `0%`；GPU1 约 `277MiB`、利用率 `0%`。
- 训练进度：日志末尾 `59997/60000`、`59998/60000`、`59999/60000`、`60000/60000`。
- 最新严格 `Total success rate`：
  - `0.03852278162057629`
  - `0.038522713176418026`
  - `0.03852285693106767`
  - `0.03852339627786232`
  - `0.03852343392255053`
- checkpoint 列表末尾：
  - `agent_55500.pt 20622806 2026-06-05 10:26:18`
  - `agent_56000.pt 20622806 2026-06-05 10:36:11`
  - `agent_56500.pt 20622806 2026-06-05 10:46:06`
  - `agent_57000.pt 20622806 2026-06-05 10:56:00`
  - `agent_57500.pt 20622806 2026-06-05 11:05:54`
  - `agent_58000.pt 20622806 2026-06-05 11:15:51`
  - `agent_58500.pt 20622806 2026-06-05 11:25:45`
  - `agent_59000.pt 20622806 2026-06-05 11:35:38`
  - `agent_59500.pt 20622806 2026-06-05 11:45:30`
  - `agent_60000.pt 20622806 2026-06-05 11:55:22`
- `best_agent.pt 20622696 2026-06-05 09:36:26`，时间戳与 `agent_53000.pt` 对齐。
- 同步尝试：
  - 批量 `scp` 因传输过慢/卡住中断，曾留下半截 `best_agent.pt`，已删除。
  - 改用单文件 `rsync -az` 同步 `best_agent.pt`。
- 本地同步结果：
  - `remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`
  - `remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt -> best_agent.pt`
  - `remote-results/teacher-cubefallfix-low37000-20260604-1530/config/b1z1_pickmulti.yaml`
- 本地/远端 `best_agent.pt` SHA256 一致：
  - `adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c`
<!-- source: session 2026-06-07 21:11-21:34 CST +0800, official-align teacher restart -->

官方环境/teacher 对齐后重训原始记录摘要：

- 用户要求：回到官方环境/teacher 对齐后重训。
- 本地 W&B 成功 run 保存文件仍存在：
  - `/tmp/wandb_97yfb8x1_b1z1_pickmulti.py`
  - `/tmp/wandb_97yfb8x1_b1z1_pickmulti.yaml`
  - `/tmp/wandb_97yfb8x1_train_multistate.py`
- `train_multistate.py` 对比：`diff -u /tmp/wandb_97yfb8x1_train_multistate.py Visual-Whole-Body-Control/high-level/train_multistate.py` 无输出，未发现有效差异。
- 本次对齐的 `high-level/envs/b1z1_pickmulti.py` 关键逻辑：
  - 初始物体 z：`cube_start_pose.p.z = table_pos[-1] + obj_height`
  - 桌高随机范围：`torch_rand_float(-0.25, 0.35, ...)`
  - 固定桌高分支：`table_heights_fix - self.table_dimz`
  - 桌面 root z 更新：`self._table_root_states[env_ids, 2] += rand_heights.squeeze(1)`
  - `_reset_actors()` 顺序：先 `super()._reset_actors(env_ids)`，再 `_reset_table(env_ids)`、`_reset_objs(env_ids)`。
- 保留的非官方但低风险兼容点：
  - `category_indices(...)` 支持小规模 smoke，不改变 10240 env 训练主路径。
  - `torch.arange` 替代官方保存代码中的 `torch.range`，避免现代 PyTorch 兼容问题。
  - 空 `env_ids` guard 避免边界崩溃。
- YAML 对齐：`low_policy_path: "data/low_policy/publiccheckrollrew_37000.pt"`。
- 本地语法验证：`python3 -m py_compile Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py` 退出码 `0`。
- 远端同步：
  - `scp ... b1z1_pickmulti.py ubuntu@<remote-host>:/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - `scp ... b1z1_pickmulti.yaml ubuntu@<remote-host>:/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
- 远端验证：
  - `python3 -m py_compile envs/b1z1_pickmulti.py` 退出码 `0`
  - 远端低层 `data/low_policy/publiccheckrollrew_37000.pt` 大小约 `2.0M`
  - SHA256：`b61e2167ea3c370668d17bbd35b641703313a0f6bc57e1dc5117c5073af69fd6`
- teacher 启动脚本改进：
  - 文件：`remote-run/remote/50_start_teacher_tmux.sh`
  - 新增 `set +e`、`${PIPESTATUS[0]}`、`Teacher training python exit code` 记录，并以原 Python exit code 退出。
  - 本地 `bash -n` 和远端 `bash -n` 均通过。
- Full-scale smoke：
  - 命令环境：`TEACHER_SMOKE_ENVS=10240 TEACHER_SMOKE_TIMESTEPS=24 VBC_USE_WANDB=0`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260607-211143.log`
  - 结果：退出码 `0`，完成 `24/24`。
- 新 teacher 长训：
  - 启动命令环境：`TEACHER_RUN_NAME=teacher-officialalign-low37000-20260607-2116 TEACHER_GPU=1 TEACHER_TMUX_SESSION=vbc_teacher_g1 TEACHER_TIMESTEPS=60000 VBC_USE_WANDB=0`
  - 启动输出：`Started teacher training in tmux session: vbc_teacher_g1`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-officialalign-low37000-20260607-2116.teacher.log`
  - 2026-06-07 21:25 CST 复查：tmux 和 `python train_multistate.py` 进程存在，GPU1 占用约 `5779MiB`，利用率 `92%`。
  - 2026-06-07 21:34:59 CST 复查：进度约 `515/60000`，最新严格 `Total success rate=5.432420686657975e-05`，`agent_500.pt` 已生成。

<!-- source: session 2026-06-08 23:40 - 2026-06-09 11:43 CST +0800, table/reset ablation training -->

table/reset 短 ablation 训练原始记录摘要：

- 本地代码改动目标：保留官方 `cube_falls` 与低层 `publiccheckrollrew_37000.pt`，只恢复上一轮较强 cubefallfix 的 table/reset 逻辑：
  - `_create_extra()` 物体初始 z：`cube_start_pose.p.z = self.table_heights[i] + obj_height`
  - `_reset_table()` 随机范围：`torch_rand_float(0, 0.5, ...)`
  - 固定桌高分支：`table_heights_fix - self.table_dimz / 2`
  - 桌面 root z：`= rand_heights.squeeze(1) - self.table_dimz / 2.0`
  - `_reset_actors()` 顺序：`_reset_table()`、`_reset_objs()`、`super()._reset_actors(env_ids)`
- 本地验证：`python3 -m py_compile high-level/envs/b1z1_pickmulti.py` 退出码 `0`。
- 同步策略：只用 `scp` 精确同步 `high-level/envs/b1z1_pickmulti.py`，没有使用带 `--delete` 的全量 `rsync`，避免误删远端已有 `publiccheckrollrew_37000.pt`。
- 远端 smoke：
  - `TEACHER_SMOKE_ENVS=10240`
  - `TEACHER_SMOKE_TIMESTEPS=24`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260608-234017.log`
  - 退出码：`0`
- 短训：
  - run：`teacher-tablereset-ablation-low37000-20260608-2341`
  - tmux：`vbc_teacher_ablate_g1`
  - GPU：`1`
  - timesteps：`10000`
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-tablereset-ablation-low37000-20260608-2341.teacher.log`
  - 2026-06-09 00:03 CST 进度约 `1086/10000`，严格成功率尾部约 `0.000919`。
  - 2026-06-09 02:59:51 CST 训练正常退出，日志含 `Teacher training python exit code: 0`。
  - 训练末尾严格 `Total success rate≈0.0195169127`。
- checkpoint：
  - `agent_1000.pt` 到 `agent_10000.pt` 均生成。
  - `agent_10000.pt` 生成于 `2026-06-09 02:59:51 CST`，大小 `20622806` bytes。
  - `best_agent.pt` 生成时间与 `agent_3000.pt` 附近对齐。
