# 实现与训练

早期没有修改 `Visual-Whole-Body-Control/` 中的项目代码，也没有执行训练。2026-05-26 已为本机仿真运行做最小配置修复，但仍未执行训练。

可用训练入口来自仓库 README：

- 低层训练：`low-level/legged_gym/scripts/train.py`
- 低层回放：`low-level/legged_gym/scripts/play.py`
- 高层 state-based teacher 训练：`high-level/train_multistate.py`
- 高层 teacher 回放：`high-level/play_multistate.py`
- 视觉 student 训练：`high-level/train_multi_bc_deter.py`
- 视觉 student 回放：`high-level/play_multi_bc_deter.py`
- 点云检查：`high-level/test_pointcloud.py`

运行高层训练前需要在配置文件中指定预训练低层策略路径。README 建议先确认低层策略可用，再用于高层策略训练。

2026-05-25 复现阶段判断：

1. 环境准备：Python 3.8、Isaac Gym、PyTorch CUDA、`rsl_rl`、`skrl`、`low-level`。当前 Python 3.8 与 Isaac Gym 基本具备，但 PyTorch CUDA 不可用、`skrl` 缺失、`rsl_rl/legged_gym` 路径指向其他项目。
2. 低层权重准备：用户已下载 `/home/hjr/projects/2-Nexus/VBC/model_38000.pt`，该 checkpoint 可被 `torch.load` 读取，`iter=38000`。这允许跳过低层从零训练，优先做低层仿真回放验证。
3. 低层仿真回放：修好环境后，使用 `low-level/legged_gym/scripts/play.py` 小规模验证低层走路与 EE goal tracking。
4. 高层 teacher 仿真：把 `high-level/data/cfg/b1z1_pickmulti.yaml` 中的 `low_policy_path` 指向 `model_38000.pt`，再用 `high-level/train_multistate.py` 训练 teacher，或用已有 teacher checkpoint 回放。当前只有低层 checkpoint，不足以直接回放完整 pick-up teacher。
5. 视觉 student 仿真：需要 teacher checkpoint，再用 `high-level/train_multi_bc_deter.py` 进行 DAgger 蒸馏。视觉 student 会启用相机渲染，GTX 1660 SUPER 6GB 只适合小规模功能验证，不适合论文级训练规模。

结论：只做仿真效果验证有条件可行，但不是当前环境下开箱即跑。低层权重已经补齐一个关键缺口；当前最大阻断项是 PyTorch CUDA 不可用、`skrl` 缺失和 Python 包路径污染。完整视觉拾取效果还需要 high-level teacher/student checkpoint 或自行训练高层策略。

2026-05-25 17:02:45 CST +0800 用户追问“如果现在只想首先在仿真环境里实现这个项目的效果，是否可行？”的补充判断：

- 可行范围：优先做低层策略仿真回放，验证 B1+Z1 的底层移动与末端目标跟踪效果。用户已有 `model_38000.pt`，因此无需先从零训练 low-level。
- 当前不可直接运行的原因：`legged-nexus-py38` 环境中 PyTorch 仍看不到 CUDA，`skrl` 缺失，`rsl_rl/legged_gym` 导入路径指向其他项目。
- 完整效果边界：仅有低层权重还不足以直接复现高层 pick-up 或 vision student 效果；需要 high-level teacher/student checkpoint，或先用低层权重支撑高层 teacher 训练，再蒸馏 student。

2026-05-25 17:13:01 CST +0800 用户进一步澄清“最终仿真效果”指高层控制和低层控制都起效、在仿真里达到论文中的效果。补充判断：

- 回放层面：如果拿到完整 checkpoint，包括 low-level、high-level teacher 或 vision student，本机可尝试以小并行数运行 `play_multistate.py` 或 `play_multi_bc_deter.py`；代码在 eval/play 时会把 high-level 环境数降到 `34`。但当前环境仍需修复 CUDA、`skrl` 和路径污染问题。
- 训练层面：本机不适合从头训练到论文级最终效果。论文中 high-level teacher 训练使用 `10240` 并行环境，vision student 因视觉渲染显存开销使用 `240` 并行环境；作者在 GTX4090/GTX3090 上训练，GTX4090 上 teacher 约 `36` 小时，student 约 `48` 小时。
- 本机配置为 GTX 1660 SUPER 6GB、i5-12400F、31GiB RAM。它可以做低层回放、小规模 teacher 调试、少量环境的 student smoke test；不建议承担完整 teacher/student 训练。

2026-05-26 低层回放修复状态：

- 已把 `/home/hjr/projects/2-Nexus/VBC/model_38000.pt` 链接到 `Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt`，使低层 `play.py --exptid google_drive --checkpoint 38000` 能按原有加载逻辑找到模型。
- 已把同一权重链接到 `Visual-Whole-Body-Control/high-level/data/low_policy/model_38000.pt`。
- 已修改 `Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`：`low_policy_path` 现在为 `data/low_policy/model_38000.pt`。
- 低层 `play.py` 已短时验证到可创建 GPU PhysX 仿真、初始化环境、构建 ActorCritic，并加载 `model_38000.pt`。测试进程由 `timeout` 主动截停，未做长时间行为评估。
- high-level 入口已验证能按真实导入顺序导入本项目 `isaacgym`、`skrl`、`B1Z1PickMulti` 和 `ActorCritic`，并能从配置路径读取 `model_38000.pt`。

2026-05-26 高层 teacher smoke test 状态：

- 已修复 `high-level/envs/b1z1_pickmulti.py` 中一个小规模 smoke test 问题：原 `_reset_envs()` 的分类成功率统计默认 `num_envs >= 33`，当 `numEnvs < 33` 时会生成非整数空 tensor 并导致 reset 报错。现在改为按 `len(self.obj_list)` 生成并过滤越界索引，且强制 `dtype=torch.long`。
- 同文件还把 `torch.range(...)` 改为 `torch.arange(...)`，消除随机移动物体逻辑中的弃用接口。
- 高层最小环境 smoke test 已通过：把 `numEnvs` 临时设为 `4`，创建 `B1Z1PickMulti`，加载低层 `model_38000.pt`，完成 `reset()` 与 `8` 次 `step()`，退出码 `0`。
- 高层标准入口短 rollout 已通过：`train_multistate.py --debug --timesteps 8 ...` 加载低层策略并完成 `8/8` timestep，退出码 `0`。
- 高层标准入口 `--debug --timesteps 24` 可完成 rollout，但在第 `24` timestep 后进入 PPO update 时触发 CUDA OOM；这说明当前机器可用于短 rollout/调试，不适合按项目默认 PPO 配置做 teacher 训练更新。

2026-05-29 远端双 3090 训练准备：

- 用户计划提供远端双 RTX 3090 机器。判断：硬件足以支撑 teacher 训练，但项目脚本默认不是多卡训练，两张卡不会自动合并显存。
- 默认训练策略：GPU0 跑 teacher，GPU1 保留给评估、第二 seed 或 teacher checkpoint 可用后的 student。
- 已创建 `remote-run/` 启动包，目标是远端 SSH 一可用就用最少步骤完成同步、安装、smoke test 和 tmux 长训启动。
- teacher 正式训练命令由 `remote-run/remote/50_start_teacher_tmux.sh` 生成，核心参数为 `train_multistate.py --timesteps 60000 --headless --task B1Z1PickMulti --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`。
- teacher smoke test 由 `remote-run/remote/40_smoke_teacher.sh` 执行，默认临时把 `b1z1_pickmulti.yaml` 的 `numEnvs` 改为 `10240`，跑 `24` timesteps；脚本退出时恢复配置。
- student 启动脚本已准备，但必须等 teacher checkpoint 产生后设置 `STUDENT_TEACHER_CKPT`。

2026-05-30 远端 teacher smoke 与长训启动：

- 远端 teacher smoke test 已实际跑通：`TEACHER_SMOKE_ENVS=10240`、`TEACHER_SMOKE_TIMESTEPS=24`，使用 `train_multistate.py --headless --task B1Z1PickMulti ...`，退出码 `0`。
- smoke 输出显示 `Low level pretrained policy loaded!`，`Total success rate 0.0`，24/24 步完成，说明远端 3090 环境可以承接 teacher rollout。
- smoke 完成后已启动 teacher 长训，tmux 会话名 `vbc_teacher_g0`。
- 长训日志路径：`/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`。
- 这意味着当前阶段已经从“准备远端环境”推进到“远端 teacher 正在训练”。

2026-05-31 远端 teacher 训练完成：

- teacher 长训已完成 `60000/60000`。
- 训练退出时间：`2026-05-31 12:24:41 CST +0800`。
- 最终常规 checkpoint：`agent_60000.pt`。
- 最佳 checkpoint 文件：`best_agent.pt`。
- 常规 `agent_*.pt` 共 `120` 个，checkpoint 目录总占用约 `2.4G`。
- 下一步应先做 teacher checkpoint 回放/评估，再决定是否用该 teacher 启动 vision student DAgger。

2026-05-31 teacher 效果确认结果：

- `agent_60000.pt` 已用 `play_multistate.py` 做 headless eval，运行到约 `1500` eval steps 后手动停止；期间 `Total success rate` 始终为 `0.0`。
- `best_agent.pt` 不能直接用当前 `play_multistate.py` 评估，脚本从文件名解析步数时对 `best_agent.pt` 报 `ValueError: invalid literal for int() with base 10: 'agent'`。
- 由于 `best_agent.pt` 时间戳与 `agent_58000.pt` 对齐，继续用 `agent_58000.pt` 作为近似候选评估；该 eval 运行到约 `1687` eval steps，`Total success rate` 仍为 `0.0`。
- 因 teacher 效果未确认，本轮没有启动 student 训练，避免浪费远端算力。
- 关键权重已经同步回本地：`agent_60000.pt`、`agent_58000.pt`、`best_agent.pt`。

2026-05-31 17:02:46 CST +0800 追加判断：

- 已确认 `best_agent.pt` 与 `agent_58000.pt` 哈希不同；`agent_58000.pt` 不能代表真正 best。
- 已在远端用 `/tmp/best_58000.pt -> best_agent.pt` 的临时链接绕过文件名解析问题，并用临时 probe 评估真正 best。
- 真正 best 评估仍失败：400 steps、34 env、266 个新 episode，`new_success=0`，`window_success_rate=0.0`，`max_lifted_object_count=0`，最大物体抬升约 `0.109m`。
- `agent_60000.pt` 原 eval 起点 probe：400 steps、255 个新 episode，`new_success=0`，最大物体抬升约 `0.076m`。
- `agent_60000.pt` 训练起点 `x=-2.0` 对照 probe：300 steps、185 个新 episode，`new_success=0`，最大物体抬升约 `0.052m`。
- 结论：当前 teacher checkpoint 不适合作为 vision student 的专家标签来源；本轮继续不启动 student。

2026-05-31 18:58:00 CST +0800 远端与 teacher 训练指标复查：

- 远端 `ubuntu@<remote-host>` 在线；`nvidia-smi` 显示 GPU0 约 `285MiB/24GiB`、GPU1 约 `15MiB/24GiB`，无正在运行的 teacher/student/play/probe 进程。
- 远端 teacher checkpoint 目录仍为 `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints`；常规 `agent_*.pt` 共 `120` 个，总占用 `2.4G`。
- 远端与本地三份关键权重 SHA256 一致：`best_agent.pt=d140cca286eb4107348aed2b4596ee1858a88bfb7bc389ced208a1de27af8299`，`agent_58000.pt=9367999f2566fa128418b60a737c9d764b37f4363a97623d7533f6f01abb3f42`，`agent_60000.pt=1ffdbdd1bbeeb3b1e72e447964943c8e86e77cf90a1e19da9d6944025d66e4e4`。
- 远端训练 event 显示 reward 确实上升：`Reward / Total reward (mean)` 首值 `0.19169`、末值 `6.68606`、峰值 `8.10409`；`Reward / Instantaneous reward (mean)` 首值 `0.04416`、末值 `0.35730`、峰值 `0.35859`。
- 完整训练日志中可解析 `Total success rate` 条目 `59949` 个，首值 `0.0`，末值 `0.00252265`，峰值约 `0.00401653`。结论仍是 reward 上升但成功率没有收敛。
- 训练配置与 README 命令一致使用 `--roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`、`numEnvs=10240`、`timesteps=60000`、`maxEpisodeLength=150`、`liftedSuccessThreshold=0.35`。
- 需要重点排查低层权重等价性：仓库原始配置指向 `publiccheckrollrew_42000.pt`，当前复现实验使用的是 Google Drive 下载并改名/链接的 `model_38000.pt`。该权重结构匹配，但尚不能证明行为与作者训练 teacher 时的低层策略等价。

2026-05-31 23:57:10 CST +0800 按作者低层环境重新训练 low-level：

- 新增远端低层脚本：
  - `remote-run/remote/90_install_low_level_env.sh`：安装 `vbc-low-cu113`。
  - `remote-run/remote/91_smoke_low_level.sh`：低层 smoke test。
  - `remote-run/remote/92_start_low_level_tmux.sh`：启动低层长训。
  - `remote-run/remote/93_monitor_low_level.sh`：监控低层长训。
  - `remote-run/remote/94_prepare_teacher_from_low.sh`：将低层 `model_42000.pt` 准备为高层 `publiccheckrollrew_42000.pt`。
  - `remote-run/remote/95_wait_low_then_start_teacher.sh` 和 `96_start_wait_low_then_teacher_tmux.sh`：等待低层 `model_42000.pt` 出现后自动启动 teacher。
- 远端低层环境已安装并验证：Python `3.8.18`，`torch 1.10.0+cu113`，CUDA `11.3`，`numpy 1.23.5`，`wandb 0.16.2`，`isaacgym 1.0rc3`，`rsl-rl 1.0.2`。
- 低层 smoke test 已完成：`LOW_SMOKE_ITERATIONS=2`，`train.py --debug --max_iterations 2 ...` 成功跑完两次 PPO iteration。
- 低层长训当前 run：`publiccheckrollrew-retrain-20260531-2354`。
- 启动参数：`LOW_NUM_ENVS=6144`，`LOW_MAX_ITERATIONS=45000`，`LOW_GPU=0`，`LOW_WANDB_MODE=disabled`。
- 当前远端状态：tmux `vbc_low_g0` 正在训练；`vbc_wait_low_teacher` 正在等待 `model_42000.pt`，准备在生成后于 GPU1 启动 teacher run `teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`。

2026-06-02 21:03:36 CST +0800 低层续训第二次恢复：

- 10:50 CST 时低层从 `model_23400.pt` 续训正常，已到 `Learning iteration 25463/45000`。
- 20:56-21:03 CST 复查发现低层 tmux 曾在 17:54 前后停止，watcher 日志记录 `model_42000.pt` 缺失且 `vbc_low_g0` 不运行；teacher 未启动。
- 低层日志末尾停在 `Learning iteration 34529/45000`，最新可用 checkpoint 为 `model_34400.pt`，没有看到新的 `model_42000.pt`。
- 已从 `model_34400.pt` 恢复：`LOW_RESUME_RUN_NAME=publiccheckrollrew-retrain-20260531-2354`、`LOW_RESUME_CHECKPOINT=34400`、`LOW_MAX_ITERATIONS=10600`，目标总步仍为 `45000`。
- 恢复后确认 tmux `vbc_low_g0` 运行，当前日志到 `Learning iteration 34415/45000`，GPU0 约 `9610MiB/24576MiB`、`93%`。
- watcher tmux `vbc_wait_low_teacher` 已重新启动，继续等待 `model_42000.pt`；出现后会准备高层低层权重并在 GPU1 启动 teacher run `teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`。

2026-06-03 08:44:59 CST +0800 低层完成并启动新 teacher：

- 低层重训已完成到 `45000/45000`：`model_45000.pt` 生成于 `2026-06-03 05:07:40 CST`，日志显示 `Low-level training exited at 2026-06-03 05:07:47 CST +0800`。
- watcher 于 `2026-06-03 02:52:38 CST` 找到 `model_42000.pt`，但执行 `94_prepare_teacher_from_low.sh` 失败：`python: 未找到命令`。原因是脚本没有激活 conda 就调用 `python`。
- 已修复 `remote-run/remote/94_prepare_teacher_from_low.sh`：在运行 Python YAML 更新前调用 `activate_env`，并已同步到远端。
- 已手动补跑 `94_prepare_teacher_from_low.sh`，将低层 `model_42000.pt` 准备为高层 `data/low_policy/publiccheckrollrew_42000.pt`，并更新远端 `b1z1_pickmulti.yaml`。
- 已启动新 teacher 长训：tmux `vbc_teacher_g1`，run `teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`，GPU1，`TEACHER_TIMESTEPS=60000`，`VBC_USE_WANDB=0`。
- 08:44:59 CST 监控确认：GPU1 约 `8733MiB/24576MiB`、`94%`，teacher 进度约 `41/60000`；早期 `Total success rate` 仍为 `0.0`，当前尚未生成 checkpoint。
- 08:47:28 CST 二次确认：tmux `vbc_teacher_g1` 仍在运行，GPU1 约 `9219MiB/24576MiB`、`94%`，teacher 进度约 `177/60000`，`Total success rate` 仍为 `0.0`。
- 12:31:05 CST 再次确认：tmux `vbc_teacher_g1` 仍在运行，GPU1 约 `10193MiB/24576MiB`、`91%`，teacher 进度约 `11507/60000`；最新常规 checkpoint 为 `agent_11500.pt`，最新可见 `Total success rate` 约 `0.00845669`。
- 18:59:56 CST 再次确认：tmux `vbc_teacher_g1` 仍在运行，GPU1 约 `10193MiB/24576MiB`、`96%`，teacher 进度约 `30631/60000`；最新常规 checkpoint 为 `agent_30500.pt`，最新可见 `Total success rate` 约 `0.01028083`。
- 2026-06-04 09:12:43 CST 复查：新 teacher 已完成，tmux server 已退出，GPU 基本空闲。最终 checkpoint 为 `agent_60000.pt`，生成于 `2026-06-04 05:24:25 CST`；日志显示 `Teacher training exited at 2026-06-04 05:24:26 CST +0800`。末尾 `Total success rate` 约 `0.00582955`，效果仍未确认，不应直接启动 student。

2026-06-04 15:29:19 CST +0800 官方 `cube_falls` 口径修复后重新启动 teacher：

- 已将本地和远端 `high-level/envs/b1z1_pickmulti.py` 的 `cube_falls` 判定恢复为 W&B 成功 run 保存代码中的官方口径：`cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`。
- 已把远端高层低层策略切到 `data/low_policy/publiccheckrollrew_37000.pt`，来源是低层重训 run `publiccheckrollrew-retrain-20260531-2354` 的 `model_37000.pt`，用于匹配 W&B 成功 run `publiccheckrollrew_37000_2`。
- 修复后 full-scale teacher smoke test 已通过：`10240` env、`24` timesteps、退出码 `0`，日志为 `/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log`。
- 已启动新 teacher 长训：tmux `vbc_teacher_g1`，run `teacher-cubefallfix-low37000-20260604-1530`，GPU1，`TEACHER_TIMESTEPS=60000`，`VBC_USE_WANDB=0`。
- 新 teacher 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-cubefallfix-low37000-20260604-1530.teacher.log`。
- 新 teacher run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530`。
- 2026-06-04 15:29 CST 复查：训练进程仍在运行，GPU1 约 `9357MiB/24576MiB`、利用率 `92%`；日志进度约 `191/60000`，`Total success rate 0.0`，尚未生成 `agent_*.pt`。
- teacher 常规 checkpoint 保存间隔来自 `high-level/train_multistate.py`：`cfg_ppo["experiment"]["checkpoint_interval"] = 500`，所以第一个关键里程碑是 `agent_500.pt`。
- 2026-06-04 15:37 CST 复查：新 teacher 已到约 `610/60000`，`agent_500.pt` 已生成，大小 `20622586` bytes，时间 `2026-06-04 15:35:41 CST`；最近 `Total success rate` 约 `0.0001717`。下一关键里程碑更新为 `agent_1000.pt`。

2026-06-05 13:53:03 CST +0800 修复后 teacher 完成与候选权重同步：

- `teacher-cubefallfix-low37000-20260604-1530` 已完成 `60000/60000`；远端 `tmux` 无训练会话，GPU 空闲。
- 最终常规 checkpoint：`agent_60000.pt`，生成时间 `2026-06-05 11:55:22 CST`。
- 最佳 checkpoint：`best_agent.pt`，生成时间 `2026-06-05 09:36:26 CST`，与 `agent_53000.pt` 时间戳对齐。
- 训练日志末尾 `Total success rate` 约 `0.0385234`。相比上一轮 `teacher-after-publiccheckrollrew-retrain-20260531-2354-42000` 的 `~0.00582955` 明显改善，但仍不是 W&B 成功 run 的量级。
- 已同步候选 `best_agent.pt` 回本地：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`。
- 本地/远端 `best_agent.pt` SHA256 一致：`adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c`。
- 本地创建 `best_53000.pt -> best_agent.pt`，用于绕过 `best_agent.pt` 文件名解析 bug。
- 本次 run 配置已同步到：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/config/b1z1_pickmulti.yaml`。

2026-06-06 10:25:15 CST +0800 student 训练启动前决策：

- 用户决定先用当前最佳候选 teacher 推进 vision student 训练，不等待下一轮官方环境完全对齐 teacher。
- 已登记待定任务：后续若追求更接近 W&B/论文成功曲线，先恢复 W&B 成功 run 保存的 `high-level/envs/b1z1_pickmulti.py` 桌面/物体 reset 逻辑，并考虑对齐 high-level 依赖版本，然后重跑 teacher。
- student teacher 选择：使用 `teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`，不要直接替换为 `agent_53000.pt`；两者时间戳接近但 SHA256 不同。
- 计划路径：先在远端创建 `best_53000.pt -> best_agent.pt` symlink 并跑短 student smoke；smoke 通过后用 `remote-run/remote/70_start_student_tmux.sh` 启动完整 student。

2026-06-06 11:52:44 CST +0800 student smoke 修复与长训启动：

- 已在远端创建 `high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt -> best_agent.pt`。
- 已修改 `Visual-Whole-Body-Control/high-level/train_multi_bc_deter.py`：student 创建环境时设置 `cfg["enableCameraSensors"] = True`，并在 `graphics_device_id < 0` 时设为 `0`。该改动只影响 student 视觉入口，不改变 teacher 训练入口。
- 已修改 `remote-run/remote/70_start_student_tmux.sh`：增加 `STUDENT_DISPLAY` 支持，并在 tmux 命令中导出 `DISPLAY`。
- 首次 student smoke 失败：不设置 `enableCameraSensors` 时 headless camera handle 为 `-1`，`_get_camera_obs()` 中 `torch.stack(...)` 收到 `NoneType`。
- 第二次 student smoke 失败：启用 `enableCameraSensors` 但不带 `DISPLAY` 时，Isaac Gym camera sensors 在初始化阶段段错误。
- 诊断验证：Isaac Gym 自带 `multiple_camera_envs.py` 不带 `DISPLAY` 会 core dump；`DISPLAY=:0` 可正常运行。
- 修复后 smoke 已通过：`DISPLAY=:0`、GPU0、`timesteps=24`、teacher `best_53000.pt`，退出码 `0`，日志 `/home/ubuntu/vbc-remote/remote-logs/student-smoke-display0-gpu0-24step-20260606-1142.log`。
- 已启动完整 student 长训：tmux `vbc_student_g0`，run `student-best53000-display0-20260606-1145`，GPU0，`timesteps=60000`，W&B disabled，teacher `best_53000.pt`。
- 2026-06-06 11:52 CST 复查：student 仍在运行，日志最新进度约 `590/60000`，GPU0 约 `23240MiB/24576MiB`。student checkpoint 保存间隔为 `1000` steps，当前尚未到首个 checkpoint。

2026-06-06 13:02:40 CST +0800 student 长训进度复查：

- tmux `vbc_student_g0` 仍在运行，训练进程为 `python train_multi_bc_deter.py ... --timesteps 60000 ... --teacher_ckpt_path .../best_53000.pt ...`。
- GPU0 采样状态为 `23243MiB/24576MiB`、利用率 `88%`；GPU1 约 `277MiB/24576MiB`、利用率 `0%`。
- 训练日志 `/home/ubuntu/vbc-remote/remote-logs/student-best53000-display0-20260606-1145.student.log` 最新解析进度为 `4648/60000`，没有 `Student training exited` 标记。
- 已生成常规 checkpoint：`agent_1000.pt`、`agent_2000.pt`、`agent_3000.pt`、`agent_4000.pt`，单个大小 `101222050` bytes。
- 当前 `best_agent.pt` 已存在，时间戳 `2026-06-06 12:17:24 CST`，与 `agent_2000.pt` 附近对齐；仍属早期训练 checkpoint，不能据此判断最终 student 质量。

2026-06-06 18:50:40 CST +0800 student 长训异常停止：

- 远端 `tmux ls` 已无 `vbc_student_g0` 会话，`pgrep` 未发现 `train_multi_bc_deter.py` 训练进程。
- GPU0 约 `548MiB/24576MiB`、利用率 `0%`；GPU1 约 `277MiB/24576MiB`、利用率 `0%`。
- student 日志最新解析进度为 `10159/60000`，最新严格 `Total success rate` 为 `0.009483057707875563`，本次日志内解析到的峰值为 `0.20833333333333334`。该峰值是早期/局部日志指标，不代表最终可用效果。
- 日志没有 `Student training exited` 标记，最近可见进度行停在 `10159/60000`，日志文件最后修改时间约 `2026-06-06 14:39 CST`。
- 已生成 checkpoint 至 `agent_10000.pt`，生成时间 `2026-06-06 14:36:56 CST`；后续可从该文件续训。
- `train_multi_bc_deter.py` 支持 `--resume`/`--checkpoint` 续训，会在 checkpoint 存在时恢复 agent 并设置 `initial_timestep`；但当前 `remote-run/remote/common.sh` 的 `student_args()` 尚未暴露 student 续训参数，需要手动启动续训或先补脚本。

2026-06-06 20:56:29 CST +0800 student 从 `agent_10000.pt` 续训：

- 已修改本地 `remote-run/remote/common.sh`：新增 `STUDENT_CHECKPOINT` 环境变量；当该变量非空时，`student_args()` 会追加 `--checkpoint "$STUDENT_CHECKPOINT"`。
- 已修改本地 `remote-run/remote/70_start_student_tmux.sh`：启动日志新增 `Checkpoint: ...`；Python 训练命令改为先 `set +e` 执行，再用 `${PIPESTATUS[0]}` 记录 `Student training python exit code`，最后写入 `Student training exited at ...` 并以原 Python exit code 退出。
- 本地 `bash -n` 检查通过；本地和远端 `student_args()` 测试都确认会输出 `--checkpoint /tmp/agent_10000.pt`。
- 已同步上述两个脚本到远端 `/home/ubuntu/vbc-remote/remote-run/remote/`，远端 `bash -n` 也通过。
- 已启动续训：tmux `vbc_student_g0`，同一 run `student-best53000-display0-20260606-1145`，`STUDENT_DISPLAY=:0`，`STUDENT_CHECKPOINT=/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145/checkpoints/agent_10000.pt`。
- 20:56 CST 复查确认：tmux 存在，`python train_multi_bc_deter.py ... --checkpoint .../agent_10000.pt` 进程存在；GPU0 约 `23038MiB/24576MiB`、利用率 `93%`。
- 日志出现 `Resuming from checkpoint: .../agent_10000.pt`，续训 tqdm 显示 `23/50000`。这是从 checkpoint 步数 `10000` 到目标 `60000` 的剩余训练区间，不是从零重训。
- 20:59 CST 二次复查确认：tmux 仍存在，GPU0 约 `23044MiB/24576MiB`、利用率 `99%`；续训 tqdm 显示 `202/50000`，约等效 `10202/60000`。当前没有新的退出标记。
- 21:00 CST 最终复查确认：tmux 仍存在，GPU0 约 `23045MiB/24576MiB`、利用率 `99%`；续训 tqdm 显示 `279/50000`，约等效 `10279/60000`。当前没有新的退出标记。

2026-06-07 13:57:03 CST +0800 student 长训完成：

- 远端 `tmux ls` 已无训练会话，`pgrep` 未发现 `train_multi_bc_deter.py` 训练进程；GPU0/GPU1 基本空闲。
- 续训段日志显示 `50000/50000`，等效总进度 `60000/60000`。
- 改进后的启动脚本成功写出退出证据：`Student training python exit code: 0`，`Student training exited at 2026-06-07 11:39:51 CST +0800`。
- 最终 checkpoint：`agent_60000.pt`，生成时间 `2026-06-07 11:39:41 CST`，大小 `101222307` bytes。
- 当前 `best_agent.pt` 生成时间为 `2026-06-07 08:42:44 CST`，与 `agent_50000.pt` 附近对齐，大小 `101222242` bytes。
- 日志末尾严格 `Total success rate` 约 `0.015296068065938879`，日志内解析到的峰值约 `0.20833333333333334`。该指标来自训练过程，下一步仍需独立仿真评估来确认实际 student 行为。

2026-06-07 21:34:59 CST +0800 官方环境/teacher 对齐后重训启动：

- 用户决定回到“官方环境/teacher 对齐后重训”路线，原因是当前 student 独立评估整体成功率极低。
- 本地和远端 `Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py` 已对齐 W&B 成功 run `publiccheckrollrew_37000_2` 保存代码中的关键 reset/table/object 逻辑：
  - `cube_start_pose.p.z = table_pos[-1] + obj_height`
  - `_reset_table()` 桌高随机范围 `torch_rand_float(-0.25, 0.35, ...)`
  - 固定桌高分支 `table_heights_fix - self.table_dimz`
  - `self._table_root_states[env_ids, 2] += rand_heights.squeeze(1)`
  - `_reset_actors()` 顺序为 `super()._reset_actors(env_ids)` 后再 reset table/object。
- 本地和远端 `high-level/data/cfg/b1z1_pickmulti.yaml` 已切到 `low_policy_path: "data/low_policy/publiccheckrollrew_37000.pt"`。远端低层文件存在，SHA256 为 `b61e2167ea3c370668d17bbd35b641703313a0f6bc57e1dc5117c5073af69fd6`。
- 保留当前代码中的低风险兼容修复：`category_indices(...)` 支持小规模 smoke；`torch.arange` 替代官方保存代码里的 `torch.range`；空 `env_ids` guard 只避免边界崩溃。这些不改变 10240 env 官方训练主路径。
- 已改进 `remote-run/remote/50_start_teacher_tmux.sh`：teacher 训练管道现在会写出 `Teacher training python exit code`。
- 新 teacher 长训已启动：
  - run：`teacher-officialalign-low37000-20260607-2116`
  - tmux：`vbc_teacher_g1`
  - GPU：`1`
  - timesteps：`60000`
  - log：`/home/ubuntu/vbc-remote/remote-logs/teacher-officialalign-low37000-20260607-2116.teacher.log`
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-officialalign-low37000-20260607-2116`
- 2026-06-07 21:34:59 CST 早期复查：进度约 `515/60000`，`agent_500.pt` 已生成，最新严格 `Total success rate=5.432420686657975e-05`。这只是启动健康信号，不能据此判断最终 teacher 质量。
