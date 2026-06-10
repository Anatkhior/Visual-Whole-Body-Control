# 当前状态

更新时间：2026-06-10 10:12:46 CST +0800。

当前目标：为后续迁移到 Isaac Sim 做远端准备。已在远端双 RTX 3090 机器上解决 Isaac Sim 5.1.0 standalone zip 的下载问题并完成压缩包校验；安装卡在宿主 Ubuntu 20.04 的 glibc/libstdc++ 版本过旧。

当前结论：代理和下载链路已经可用；Isaac Sim 5.1.0 zip 已下载到远端 `/home/ubuntu/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64.zip`，大小 `8768419777` bytes，`unzip -tq` 通过。`/home/ubuntu/isaacsim-5.1.0` 已解压完成，目录约 `17G`。`post_install.sh` 失败于系统运行库：远端 Ubuntu `20.04.6` 的 glibc 为 `2.31`、libstdc++ 最高 `GLIBCXX_3.4.28`，而 Isaac Sim 5.1.0 的 `kit/libcarb.so` 需要 `GLIBC_2.32/2.33/2.34` 和 `GLIBCXX_3.4.29/3.4.30`。当前账号没有免密 sudo，远端也没有 Docker/NVIDIA Container Toolkit，不能在本轮直接升级系统或安装容器栈。

上一条 VBC 训练结论仍有效：table/reset ablation 比 official-align 的随机高度 probe `0/10` 有改善，但仍弱于当前最强候选 `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt`。因此暂不建议把 table/reset ablation 扩展成 `60000` 步长训，也不建议用它替代 cubefallfix best 训练 student。

项目目录：

- 工作区：`/home/hjr/projects/2-Nexus/VBC/`
- 项目：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/`
- 日志：`/home/hjr/projects/2-Nexus/VBC/agent-log/`
- 远端启动包：`/home/hjr/projects/2-Nexus/VBC/remote-run/`
- 仓库内续作入口：`REPRODUCTION_HANDOFF.md`、`agent-log/00-start-here.md`、`agent-log/01-current-state.md`

已完成：

- 低层权重存在并可读取：`/home/hjr/projects/2-Nexus/VBC/model_38000.pt`，`iter=38000`，`model_state_dict` 有 41 个键。
- 本机低层 headless 数值回放与 `.mp4` 视频证据已完成。
- 本机高层 teacher 短 rollout 已通过；`train_multistate.py --debug --timesteps 8 ...` 退出码 `0`。
- 本机 GTX 1660 SUPER 6GB 已确认不适合 teacher PPO 更新：`--debug --timesteps 24` 在 PPO update 阶段 CUDA OOM。
- 已创建 `remote-run/` 远端快速训练启动包，包含本地同步脚本、远端环境安装脚本、导入检查、teacher smoke test、tmux 长训启动、监控、student 启动和结果打包脚本。
- 远端已完成同步、环境安装、导入检查和 teacher smoke test；`TEACHER_SMOKE_ENVS=10240`、`TEACHER_SMOKE_TIMESTEPS=24` 已通过。
- 远端 teacher 长训练已完成：`60000/60000`，训练进程于 `2026-05-31 12:24:41 CST +0800` 退出。
- 最终 checkpoint 已生成：`agent_60000.pt`，并保留 `best_agent.pt`。
- 已对 `agent_60000.pt` 与 `agent_58000.pt` 做 headless teacher eval；二者在已跑的评估窗口内 `Total success rate` 均为 `0.0`，teacher 效果未确认。
- `best_agent.pt` 不能直接被 `play_multistate.py` 读取，原因是脚本从 checkpoint 文件名最后一个 `_` 后解析步数，`best_agent.pt` 会解析出 `agent` 并触发 `ValueError`。
- 已绕过该文件名问题，将远端 `best_agent.pt` 临时链接为 `/tmp/best_58000.pt` 并用临时探针评估真正 best 权重；400 steps、34 env、266 个新 episode，`new_success=0`，`window_success_rate=0.0`，`max_lifted_object_count=0`，最大物体抬升约 `0.109m`，仍低于配置成功阈值 `0.35m`。
- `agent_60000.pt` 也用临时探针确认失败：原 eval 起点 400 steps/255 个新 episode `new_success=0`；训练起点 `x=-2.0` 对照 300 steps/185 个新 episode `new_success=0`。
- TensorBoard 训练曲线显示 reward 有上升但成功率没有有效收敛：`Reward / Total reward (mean)` 从 `0.1917` 到 `6.6861`，`Reward / Instantaneous reward (mean)` 从 `0.0442` 到 `0.3573`，但训练末尾累计 `Total success rate` 仅约 `0.00252`。
- 低层 `model_38000.pt` 的 checkpoint 结构和 `high-level/utils/low_level_model.py` 对齐：`actor.priv_encoder.0.weight (64, 18)`、`actor.history_encoder.encoder.0.weight (30, 71)`、`actor.history_encoder.conv_layers.0.weight (20, 30, 4)`、`actor.actor_backbone.0.weight (128, 91)`、`actor.actor_leg_control_head.4.weight (12, 128)` 等关键形状都对得上。
- 低层 checkpoint 的 `infos` 字段为 `None`，所以无法从权重本身直接恢复作者训练 run 名称或额外超参，只能确认它在结构和实际回放上都可用。
- 2026-05-31 18:43 CST 复查远端：没有 `train_multistate.py`、`play_multistate.py`、`train_multi_bc_deter.py` 等后台进程；两张 RTX 3090 基本空闲；checkpoint 目录仍为 `120` 个常规 `agent_*.pt`，总占用 `2.4G`。
- 2026-05-31 18:50 CST 解析远端 TensorBoard event：`Reward / Total reward (mean)` 从 `0.19169` 到 `6.68606`，峰值 `8.10409`；`Reward / Instantaneous reward (mean)` 从 `0.04416` 到 `0.35730`，峰值 `0.35859`。完整训练日志中 `Total success rate` 共有 `59949` 个可解析条目，最终 `0.00252265`，峰值约 `0.00401653`，远低于可用 teacher 的预期。
- 当前最强可疑点不是低层 checkpoint 结构错误，而是高层期望的低层 checkpoint 与当前可用 `model_38000.pt` 可能并非同一策略：仓库原始 `b1z1_pickmulti.yaml` 指向 `/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt`，本地为运行而改成了 `data/low_policy/model_38000.pt`。该点仍需拿到作者期望权重或确认二者等价后才能排除。
- 2026-05-31 23:31 CST 继续确认 `publiccheckrollrew_42000.pt`：本地工作区和远端训练机均未找到该文件；远端只存在高层 teacher 的 `agent_42000.pt`，不是低层 checkpoint。
- W&B 公开项目 `ericonaldo/b1z1-low` 中存在 run `yf5dxfg1`，`displayName=publiccheckrollrew`，状态 `finished`，创建时间 `2024-06-21T09:56:18Z`；完整 `output.log` 确认该 run 到达 `Learning iteration 42000/45000` 和 `44999/45000`。
- 但 W&B run 的公开文件列表没有 `.pt` checkpoint；项目级 artifact type 只有 `job`、`wandb-history`、`wandb-events`，run output artifacts 是 history/events parquet，input job artifact 只含 `requirements.frozen.txt` 和 `wandb-job.json`。直接访问 `publiccheckrollrew_42000.pt` 和 `model_42000.pt` 均返回 `404`。
- 当前结论：可以确认作者的 `publiccheckrollrew` 低层训练 run 公开存在且跑过 42000 iteration；不能确认 `publiccheckrollrew_42000.pt` 已公开发布或能从 W&B/repo/远端恢复。更可能是作者本地日志目录中的 checkpoint，未作为 W&B artifact 上传。
- 已在远端创建低层专用环境 `vbc-low-cu113`，贴近 W&B 原始低层环境：Python `3.8.18`、`torch==1.10.0+cu113`、`torchvision==0.11.1+cu113`、`torchaudio==0.10.0+cu113`、`numpy==1.23.5`、`isaacgym==1.0rc3`、`rsl-rl==1.0.2`、`wandb==0.16.2`。导入检查显示 `torch.cuda.is_available() == True`。
- 低层 smoke test 已通过：远端 `LOW_SMOKE_ITERATIONS=2`，`train.py --debug --max_iterations 2 ...` 完成 `Learning iteration 0/2` 和 `1/2`，PPO update 正常。
- 低层首次长训曾因 `wandb.init(mode="online")` 联网超时退出；已修复启动脚本，长训时显式设置 `WANDB_MODE=disabled` 和 `WANDB_DISABLED=true`。
- 当前低层长训正在远端 tmux `vbc_low_g0` 中运行：`LOW_RUN_NAME=publiccheckrollrew-retrain-20260531-2354`，`LOW_NUM_ENVS=6144`，`LOW_MAX_ITERATIONS=45000`，GPU0，log 为 `/home/ubuntu/vbc-remote/remote-logs/publiccheckrollrew-retrain-20260531-2354.low.log`，run dir 为 `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354`。
- 2026-06-02 09:06 CST 复查发现低层原 tmux 和 watcher 均已停止，teacher 未启动；低层停在 `Learning iteration 23545/45000`，最新 checkpoint 为 `model_23400.pt`，生成时间 `2026-06-01 18:12:56 CST`。日志末尾没有 Python traceback、OOM、磁盘满或 `Low-level training exited` 标记，远端也没有重启，原因暂不明确。
- 2026-06-02 09:08 CST 已从 `model_23400.pt` 续训：`LOW_RESUME_RUN_NAME=publiccheckrollrew-retrain-20260531-2354`、`LOW_RESUME_CHECKPOINT=23400`、`LOW_MAX_ITERATIONS=21600`，目标总步仍为 `45000`。
- 2026-06-02 10:50 CST 续训正常运行：tmux `vbc_low_g0` 运行中，当前 `Learning iteration 25463/45000`，最新 checkpoint 为 `model_25400.pt`，生成时间 `2026-06-02 10:47:08 CST`；GPU0 约 `9615MiB/24576MiB`、`77%`。按当前速度到 `model_42000.pt` 粗估还需约 `13h` 左右，约在 `2026-06-02 23:30 CST` 附近。
- 2026-06-02 17:54 CST watcher 记录：低层 tmux `vbc_low_g0` 已停止且 `model_42000.pt` 仍缺失。复查时没有 `train.py` 进程，teacher 未启动；低层日志末尾停在 `Learning iteration 34529/45000`，最新可用 checkpoint 为 `model_34400.pt`。
- 2026-06-02 21:02 CST 已从 `model_34400.pt` 再次续训：`LOW_RESUME_RUN_NAME=publiccheckrollrew-retrain-20260531-2354`、`LOW_RESUME_CHECKPOINT=34400`、`LOW_MAX_ITERATIONS=10600`，目标总步仍为 `45000`。
- 2026-06-02 21:03 CST 续训已确认运行：tmux `vbc_low_g0` 和 watcher `vbc_wait_low_teacher` 均存在；当前低层日志到 `Learning iteration 34415/45000`，mean reward 约 `14.53`；GPU0 约 `9610MiB/24576MiB`、`93%`，GPU1 空闲。
- watcher 已重新启动：等待 `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_42000.pt`；出现后会复制到高层 `data/low_policy/publiccheckrollrew_42000.pt`、更新远端 `b1z1_pickmulti.yaml`，并在 GPU1 启动 teacher run `teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`。按约 `2.7-2.8s/iteration` 粗估，若不中断，`model_42000.pt` 可能在 `2026-06-03 02:50 CST` 左右出现，完整到 `45000` 约在 `2026-06-03 05:10 CST` 左右。
- 2026-06-03 02:52 CST watcher 找到 `model_42000.pt`，但执行 `94_prepare_teacher_from_low.sh` 时失败：`python: 未找到命令`。原因是该脚本在非交互 shell 中没有激活 conda，直接调用了 `python`。
- 低层训练继续跑完：`model_45000.pt` 已生成，时间 `2026-06-03 05:07:40 CST`；低层日志显示 `Learning iteration 44999/45000` 后于 `2026-06-03 05:07:47 CST` 正常退出。末尾 mean reward 约 `18.44`。
- 已修复并同步 `remote-run/remote/94_prepare_teacher_from_low.sh`：在更新 YAML 前调用 `activate_env`。随后手动补跑准备步骤，已把低层 `/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_42000.pt` 复制为高层 `high-level/data/low_policy/publiccheckrollrew_42000.pt`，并更新远端 `b1z1_pickmulti.yaml`。
- 2026-06-03 08:44 CST 已启动新 teacher：tmux `vbc_teacher_g1`，run `teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000.teacher.log`。2026-06-03 08:47 CST 复查时 GPU1 约 `9219MiB/24576MiB`、`94%`，进度约 `177/60000`，早期 `Total success rate` 仍为 `0.0`。
- 2026-06-03 12:31 CST 复查：teacher 仍在 tmux `vbc_teacher_g1` 中正常训练，GPU1 约 `10193MiB/24576MiB`、`91%`，进度约 `11507/60000`。已生成 `agent_11500.pt`，最新可见 `Total success rate` 约 `0.00845669`，`best_agent.pt` 时间戳为 `2026-06-03 12:10:53 CST`。
- 2026-06-03 18:59 CST 复查：teacher 仍在 tmux `vbc_teacher_g1` 中正常训练，GPU1 约 `10193MiB/24576MiB`、`96%`，进度约 `30631/60000`。已生成 `agent_30500.pt`，最新可见 `Total success rate` 约 `0.01028083`，`best_agent.pt` 时间戳为 `2026-06-03 15:33:59 CST`。
- 2026-06-04 09:12 CST 复查：tmux 已无运行会话，GPU0/GPU1 基本空闲；新 teacher 训练已完成。最终 checkpoint `agent_60000.pt` 生成于 `2026-06-04 05:24:25 CST`，`best_agent.pt` 时间戳为 `2026-06-04 04:10:51 CST`，日志显示 `Teacher training exited at 2026-06-04 05:24:26 CST +0800`。
- 新 teacher 训练末尾 `Total success rate` 约 `0.00582955`；这比早期的 `0.01028` 还低，仍明显不是一个已经确认可用的 teacher。需要先做 headless eval/探针评估，再决定是否同步权重和启动 student。
- 关键 teacher 权重已同步回本地，且可用本地 `legged-nexus-py38` 环境 `torch.load(..., map_location="cpu")` 读取。
- 本地关键权重 SHA256 已与远端核对一致：`best_agent.pt=d140cca286eb4107348aed2b4596ee1858a88bfb7bc389ced208a1de27af8299`，`agent_58000.pt=9367999f2566fa128418b60a737c9d764b37f4363a97623d7533f6f01abb3f42`，`agent_60000.pt=1ffdbdd1bbeeb3b1e72e447964943c8e86e77cf90a1e19da9d6944025d66e4e4`。
- 2026-06-04 12:35 CST 复查远端：没有 `train_multistate.py`、`train.py` 或 `vbc_teacher_probe` 训练/评估进程；GPU0 约 `543MiB/24576MiB`、GPU1 约 `277MiB/24576MiB`，利用率均 `0%`。问题不是训练仍在进行，而是 checkpoint 行为未通过。
- 远端新 teacher run 内实际配置已确认使用 `low_policy_path: data/low_policy/publiccheckrollrew_42000.pt`，不是旧的 `model_38000.pt` 路径。
- 严格过滤 `^Total success rate ` 后，新 teacher 日志共有 `59917` 条可解析成功率记录：首值 `0.0`、末值 `0.00582955`、峰值 `0.01045189`。说明训练期间最高累计成功率也只有约 `1.05%`。
- 2026-06-04 12:42 CST 重新运行 `agent_60000.pt` 定向 probe：300 steps、34 env、`new_episodes=172`、`new_success=0`、`window_success_rate=0.0`、`max_lifted_object_count=0`、`max_curr_height=0.108388`、`min_curr_dist=0.073030`。
- 同一 probe 显示平均 base-object 距离从 `0.6223m` 降到 `0.4295m`，平均 reward 从 `0.2130` 到 `0.4792`；动作统计中 gripper 维度后期均值转负，说明策略会靠近并尝试闭爪，但没有把物体抬到 `liftedSuccessThreshold=0.35m`。
- 当前最强判断：失败模式是“接近/部分闭爪/少量抬动物体，但没有稳定抓取和高抬升”。这比“训练未完成、路径没切换、best 文件名解析 bug、eval 起点错误”更符合现有证据。
- 2026-06-04 13:03 CST 对比 W&B 官方高层项目：成功 run 如 `publiccheckrollrew_37000_2` 的 `Reward / Total reward (mean)` 约 `93.5`、多类别成功率约 `0.5-0.88`；本次新 teacher TensorBoard 末值仅 `8.256`、峰值 `9.513`，和成功 run 不在同一量级。
- termination probe 显示本次 `agent_60000.pt` 的 172 次 reset 中 `cube_falls=170`、`ik_fail=2`、`timeout=0`；策略把末端目标 z 快速压低到约 `0.12-0.19` 并在低位闭爪，主要导致物体掉落而不是抬升。
- 2026-06-04 15:13 CST 从 W&B 成功 run `publiccheckrollrew_37000_2` 下载保存的 `data/cfg/b1z1_pickmulti.yaml`、`envs/b1z1_pickmulti.py`、`train_multistate.py` 并对比本地代码：配置除低层路径外基本一致；关键差异是官方成功 run 的 `cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`，当前本地/远端代码为更严格的 `cube_falls = z_cube < self.table_heights`。
- 该差异直接对应当前 probe 中 `cube_falls` 大量 reset 的现象。下一步建议先恢复官方 `cube_falls` 判定，再用低层 `publiccheckrollrew_37000.pt` 跑短 probe/短训验证，而不是继续沿用当前 reset 判定重训。
- 2026-06-04 15:29 CST 已恢复本地与远端 `high-level/envs/b1z1_pickmulti.py` 中的官方 `cube_falls` 判定：`cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`。
- 远端高层配置已切到 `low_policy_path: data/low_policy/publiccheckrollrew_37000.pt`；该文件由低层重训 run 的 `model_37000.pt` 复制得到，用来匹配 W&B 成功 run `publiccheckrollrew_37000_2` 的低层步数。
- 修复后 termination probe 用旧失败 teacher `agent_60000.pt` 做 300-step 对照：`new_episodes=10`、`new_success=0`、`reset=10`、新官方口径 `cube_falls=6`、旧严格口径 `cube_below_table=163`、`ik_fail=3`。这说明 reset 行为已明显从“几乎每次低于桌面即 reset”改善为官方容差口径，但旧 teacher 本身仍不可用。
- 修复后 full-scale teacher smoke 已通过：10240 env、24 timesteps、退出码 `0`，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log`。
- 已启动修复后新 teacher 长训：tmux `vbc_teacher_g1`，run `teacher-cubefallfix-low37000-20260604-1530`，GPU1，`60000` timesteps，W&B disabled，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-cubefallfix-low37000-20260604-1530.teacher.log`。
- 2026-06-04 15:29 CST 复查：训练进程 `python train_multistate.py ...` 正在运行；GPU1 约 `9357MiB/24576MiB`、利用率 `92%`；日志进度约 `191/60000`，`Total success rate` 仍为 `0.0`，尚未到首个常规 checkpoint 保存点。
- 2026-06-04 15:37 CST 复查：训练进度约 `610/60000`，首个常规 checkpoint `agent_500.pt` 已生成，大小 `20622586` bytes，生成时间 `2026-06-04 15:35:41 CST`；最近 `Total success rate` 约 `0.0001717`。这只是早期非零信号，还不能证明收敛。
- 2026-06-04 15:40 CST 复查：训练进度约 `749/60000`，GPU1 约 `10209MiB/24576MiB`、利用率 `93%`；最近 `Total success rate` 约 `0.0002670`。仍处于很早期，应继续等 `agent_1000.pt` 和更长曲线。
- 2026-06-05 12:32 CST 复查：修复后 teacher 已完成 `60000/60000`，远端 GPU0/GPU1 空闲；`agent_60000.pt` 生成于 `2026-06-05 11:55:22 CST`，`best_agent.pt` 生成于 `2026-06-05 09:36:26 CST`，时间戳与 `agent_53000.pt` 对齐。
- 训练日志末尾严格 `Total success rate` 约 `0.0385234`，明显高于上一轮 `~0.00583`，但仍远低于 W&B 成功 run 的大致 `0.4-0.9` 成功率区间。
- `agent_60000.pt` 300-step termination probe：`new_episodes=40`、`new_success=5`、窗口成功率 `0.125`、`lifted_object=5`、`cube_falls=35`、`ik_fail=0`。最终权重已有有效拾取，但不强。
- `best_agent.pt` 不能直接被 `train_multistate.py`/probe 解析，仍会触发 `ValueError: invalid literal for int() with base 10: 'agent'`；已用 `/tmp/best_53000.pt -> best_agent.pt` 绕过，因为 `best_agent.pt` 时间戳与 `agent_53000.pt` 对齐。
- `best_agent.pt` 300-step probe：`new_episodes=12`、`new_success=4`、窗口成功率 `0.3333`、`lifted_object=4`、`cube_falls=8`、`ik_fail=0`。
- `best_agent.pt` 1000-step probe：`new_episodes=23`、`new_success=7`、窗口成功率 `0.30435`、`lifted_object=7`、`cube_falls=16`、`ik_fail=0`。这是目前最强候选 teacher 证据。
- 已同步候选权重回本地：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`，本地/远端 SHA256 均为 `adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c`。
- 本地还创建了 `best_53000.pt -> best_agent.pt` symlink，用于绕过后续评估脚本的 checkpoint 文件名解析 bug；本次 run 配置也已同步到 `remote-results/teacher-cubefallfix-low37000-20260604-1530/config/b1z1_pickmulti.yaml`。
- 已创建综合实验日志：`/home/hjr/projects/2-Nexus/VBC/teacher_success_log.md`，覆盖从项目克隆、低层验证、远端 teacher 失败诊断、低层重训、`cube_falls` 修复，到当前候选 `best_agent.pt` 的训练与 probe 证据。
- 2026-06-05 23:46 CST 对比 W&B 成功 run `publiccheckrollrew_37000_2` 保存文件后确认：YAML 结构化差异只有 `low_policy_path` 的绝对/相对路径；`train_multistate.py` 无有效差异；但 `high-level/envs/b1z1_pickmulti.py` 仍有高影响环境差异，包括物体初始 z、桌面高度随机分布、固定桌高分支、桌面 root z 更新方式和 `_reset_actors` 调用顺序。
- 当前更准确的解释：本轮 teacher 已因 `cube_falls` 修复而明显改善，但仍低于官方，最可能是“环境动力学/任务分布没有完全对齐 + 低层 checkpoint 不是作者隐藏的原始 `publiccheckrollrew_37000.pt` + 高层依赖版本不同”的叠加结果，不应只归因于训练随机性。
- 2026-06-06 10:25 CST 用户决定先使用当前最佳 teacher 推进 student。已把“对齐官方 `b1z1_pickmulti.py` reset 逻辑和 high-level 依赖后重跑 teacher”登记为待定任务。
- 远端复查：GPU0/GPU1 均空闲，无 `train_multistate.py`、`train_multi_bc_deter.py`、play 进程；当前最佳 teacher 文件存在于远端 `teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`。
- 注意：`best_agent.pt` 与同时间戳的 `agent_53000.pt` SHA256 不同，不能把 `agent_53000.pt` 视为最佳权重等价替代。student 应使用 `best_agent.pt` 本体，或使用指向它的数字后缀 symlink `best_53000.pt`。
- 2026-06-06 10:28 CST 已在远端创建 `best_53000.pt -> best_agent.pt`，用于给 student 的 `--teacher_ckpt_path` 一个数字后缀路径，同时实际加载当前最佳 `best_agent.pt`。
- student 初次 smoke 暴露两个相机问题：未设置 `enableCameraSensors` 时 headless 下 camera handle 为 `-1` 并在 `_get_camera_obs()` 处拿到 `None`；设置该项后，若 SSH 会话不绑定远端 X display，会触发 Isaac Gym camera sensors 段错误。
- 已做最小代码修复：`high-level/train_multi_bc_deter.py` 在创建 student 环境时设置 `cfg["enableCameraSensors"] = True`，并在未指定 `graphics_device_id` 时默认用 `0`。该修复已同步到远端。
- 已验证 Isaac Gym 自带 `multiple_camera_envs.py`：不带 `DISPLAY` 会 core dump，显式 `DISPLAY=:0` 可正常创建 camera handles。结论：远端 student 视觉训练必须绑定现有 X display `:0`。
- 已修改并同步 `remote-run/remote/70_start_student_tmux.sh`，支持 `STUDENT_DISPLAY` 并在日志中写出 `Display: :0`。
- student smoke 已通过：`DISPLAY=:0`、GPU0、`timesteps=24`、teacher `best_53000.pt`，日志 `/home/ubuntu/vbc-remote/remote-logs/student-smoke-display0-gpu0-24step-20260606-1142.log`，退出码 `0`。
- 完整 student 长训已启动：tmux `vbc_student_g0`，run `student-best53000-display0-20260606-1145`，GPU0，`STUDENT_TIMESTEPS=60000`，`STUDENT_DISPLAY=:0`，teacher `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt`。
- 2026-06-06 11:52 CST 复查：`vbc_student_g0` 仍在运行，进程为 `python train_multi_bc_deter.py ...`；GPU0 约 `23240MiB/24576MiB`，利用率采样 `19%`；日志最新解析进度约 `590/60000`，尚未到首个 student checkpoint 保存点 `1000`。
- 2026-06-06 13:02 CST 复查：`vbc_student_g0` 仍在运行，进程为 `python train_multi_bc_deter.py ...`；GPU0 约 `23243MiB/24576MiB`、利用率采样 `88%`，GPU1 空闲；日志最新解析进度 `4648/60000`，尚无退出标记。已生成 `agent_1000.pt`、`agent_2000.pt`、`agent_3000.pt`、`agent_4000.pt`，单个约 `101222050` bytes；`best_agent.pt` 当前时间戳为 `2026-06-06 12:17:24 CST`，与 `agent_2000.pt` 附近对齐。
- 2026-06-06 18:50 CST 复查：`vbc_student_g0` 不存在，远端无 `train_multi_bc_deter.py` 进程；GPU0/GPU1 空闲。student 日志最新解析进度为 `10159/60000`，没有 `Student training exited` 正常退出标记；最新 checkpoint 为 `agent_10000.pt`，生成时间 `2026-06-06 14:36:56 CST`。日志中未检出 traceback/OOM/segfault/error，远端 uptime 显示未重启，磁盘 `/` 仍有约 `118G` 可用；停止原因暂不明确。
- 2026-06-06 20:55 CST 已修复并同步 student 启动脚本：`remote-run/remote/common.sh` 增加 `STUDENT_CHECKPOINT`，`remote-run/remote/70_start_student_tmux.sh` 会记录 checkpoint 和 Python 退出码。
- 2026-06-06 20:56 CST 复查：已从 `agent_10000.pt` 续训同一 run；`vbc_student_g0` 重新存在，`train_multi_bc_deter.py` 进程带 `--checkpoint .../agent_10000.pt`；GPU0 约 `23038MiB/24576MiB`、利用率 `93%`，GPU1 空闲。日志有 `Resuming from checkpoint: .../agent_10000.pt`；续训进度条显示 `23/50000`，可理解为从 `10000` 起继续到 `60000` 的剩余训练段。
- 2026-06-06 20:59 CST 二次复查：`vbc_student_g0` 仍在运行，GPU0 约 `23044MiB/24576MiB`、利用率 `99%`；续训进度条显示 `202/50000`，约等效 `10202/60000`。目前没有新的退出标记。
- 2026-06-06 21:00 CST 最终复查：`vbc_student_g0` 仍在运行，GPU0 约 `23045MiB/24576MiB`、利用率 `99%`；续训段显示 `279/50000`，约等效 `10279/60000`。目前没有新的退出标记。
- 2026-06-07 13:57 CST 复查：student 续训已完成，远端无 tmux 会话、无 `train_multi_bc_deter.py` 训练进程，GPU0/GPU1 空闲。日志显示续训段 `50000/50000`，等效总进度 `60000/60000`，并写出 `Student training python exit code: 0` 与 `Student training exited at 2026-06-07 11:39:51 CST +0800`。
- 最终 student checkpoint 已生成：`agent_60000.pt`，生成时间 `2026-06-07 11:39:41 CST`，大小 `101222307` bytes；`best_agent.pt` 生成时间为 `2026-06-07 08:42:44 CST`，与 `agent_50000.pt` 附近对齐，大小 `101222242` bytes。
- student 训练日志末尾严格 `Total success rate` 约 `0.0152961`，日志内解析到的 `Total success rate` 峰值约 `0.2083333`。这只是训练过程指标，尚不能替代独立仿真评估。
- 2026-06-07 20:29 CST 评估复查：远端无 `train_multi_bc_deter.py`、`play_multi_bc_deter.py` 或 `train_multistate.py` 进程；GPU0 约 `548MiB/24576MiB`、GPU1 约 `277MiB/24576MiB`，利用率均 `0%`。
- `agent_60000.pt` headless 限时评估：日志 `/home/ubuntu/vbc-remote/remote-logs/student-eval-agent60000-headless420s-20260607-1421.log`，本地已同步为 `remote-results/student-best53000-display0-20260606-1145/logs/student-eval-agent60000-headless420s-20260607-1421.log`；运行到约 `1322/14000`，外层 `timeout` 退出码 `124`。最新 `Total success rate=0.0167130919`，窗口峰值 `0.0208333333`；类别记录显示成功只来自 `Bowl`，最新 `Bowl=0.0869565`，其它类别为 `0.0`。
- `best_agent.pt` 通过远端 `best_50000.pt -> best_agent.pt` symlink 评估：日志 `/home/ubuntu/vbc-remote/remote-logs/student-eval-best50000-headless420s-20260607-1502.log`，本地已同步为 `remote-results/student-best53000-display0-20260606-1145/logs/student-eval-best50000-headless420s-20260607-1502.log`；运行到约 `1300/14000`，外层 `timeout` 退出码 `124`。最新 `Total success rate=0.0105540897`，窗口峰值 `0.0192307692`；类别记录同样只有 `Bowl` 非零，最新 `Bowl=0.0238095`，其它类别为 `0.0`。
- 当前独立评估结论：`agent_60000.pt` 在该限时窗口内略强于 `best_agent.pt`，但二者整体成功率都极低，不能作为可展示的论文复现结果。
- 视频录制评估尚未成功：`--record_video` 路径因远端 `vbc-py38` 缺少 `imageio` 失败，日志已同步到 `remote-results/student-best53000-display0-20260606-1145/logs/student-eval-agent60000-record150-20260607-1402.log`。这是录制依赖问题，不是策略本身的失败证据。
- 本地已同步 student 训练日志、两份 headless 评估日志、失败视频录制日志和 run 配置，且日志 SHA256 与远端一致。student 大权重本地同步未完成：远端到本地单文件 `rsync` 只有约 `10-20KiB/s`，预估需数小时；已终止并清理不完整本地 checkpoint。远端权重仍完整存在，`agent_60000.pt` SHA256 为 `fd7f1e0cf0406ed41b02b3e52737a9bc340f5a786d0329b21cd5d0167ac346c7`，`best_agent.pt` SHA256 为 `053e21f8a94fa6acdac1b30854350ff71a3a0bc63cf6783bb2674e6e19e2d69e`。
- 2026-06-07 21:11 CST 起执行官方环境/teacher 对齐后重训路线：本地和远端 `high-level/envs/b1z1_pickmulti.py` 已对齐 W&B 成功 run 保存代码的关键逻辑：
  - 初始物体 z：`cube_start_pose.p.z = table_pos[-1] + obj_height`
  - 桌高随机范围：`torch_rand_float(-0.25, 0.35, ...)`
  - 固定桌高分支：`table_heights_fix - self.table_dimz`
  - 桌面 root z 更新：`+= rand_heights.squeeze(1)`
  - `_reset_actors()` 调用顺序：先 `super()._reset_actors(env_ids)`，再 `_reset_table()`、`_reset_objs()`
  - 保留小规模 smoke 兼容修复：`category_indices(...)` 和 `torch.arange`，这些不改变 10240 env 官方训练主路径。
- 本地和远端高层配置已切到 `low_policy_path: "data/low_policy/publiccheckrollrew_37000.pt"`；远端文件存在，大小约 `2.0M`，SHA256 `b61e2167ea3c370668d17bbd35b641703313a0f6bc57e1dc5117c5073af69fd6`。
- 对齐后 full-scale teacher smoke test 已通过：`TEACHER_SMOKE_ENVS=10240`、`TEACHER_SMOKE_TIMESTEPS=24`，退出码 `0`，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260607-211143.log`。
- 已改进 `remote-run/remote/50_start_teacher_tmux.sh`：teacher 训练管道现在会记录 `Teacher training python exit code`，避免异常退出时缺少退出证据。
- 新 teacher 长训已启动：run `teacher-officialalign-low37000-20260607-2116`，tmux `vbc_teacher_g1`，GPU1，`60000` timesteps，W&B disabled，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-officialalign-low37000-20260607-2116.teacher.log`。
- 2026-06-07 21:38:17 CST 复查：训练进度约 `689/60000`，最新严格 `Total success rate=0.00018343762101787498`，GPU1 约 `9429MiB/24576MiB`、利用率 `91%`；首个 checkpoint `agent_500.pt` 已生成，大小约 `20M`，路径 `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-officialalign-low37000-20260607-2116/checkpoints/agent_500.pt`。
- 2026-06-07 22:00 CST 已将续作所需的实验记录整理进 `Visual-Whole-Body-Control` Git 仓库：新增 `agent-log/`、`remote-run/`、`teacher_success_log.md`、`REPRODUCTION_HANDOFF.md`。出于公开仓库安全考虑，未纳入 `remote-run/local/remote.env`、`agent-log/SSH.md`、`remote-results/`、checkpoint、日志大文件和视频；仓库内记录已将远端 IP 与本机 SSH key 路径替换为占位符。
- 2026-06-07 22:21 CST 继续整理仓库内交接资料：`REPRODUCTION_HANDOFF.md` 已中文化；`remote-run/local/sync_to_remote.sh` 不再强制要求仓库外层存在 `model_38000.pt`，改为可选 `LOW_POLICY_LOCAL_PATH` 上传；`remote-run/remote/20_prepare_project.sh` 支持 `LOW_POLICY_TARGET_NAME`，可直接准备 `publiccheckrollrew_37000.pt` 等非默认低层权重名。
- 2026-06-07 22:27 CST 本地 Git 提交已创建在分支 `repro-handoff-and-training-fixes`，提交信息为 `chore: add reproduction handoff and remote training scripts`。尝试推送到 `origin` 失败，GitHub 返回 `fatal: could not read Username for 'https://github.com'`；这表示当前机器没有可用 HTTPS 凭据或写权限，后续需要配置 GitHub token/凭据，或把分支推到用户自己的 fork。
- 2026-06-08 09:19 CST 用户已将临时 SSH 公钥加入 GitHub 后，已通过 SSH 成功推送分支 `repro-handoff-and-training-fixes` 到 `git@github.com:Anatkhior/Visual-Whole-Body-Control.git`。GitHub 返回 PR 创建地址：`https://github.com/Anatkhior/Visual-Whole-Body-Control/pull/new/repro-handoff-and-training-fixes`。
- 2026-06-08 09:58 CST 复查新 teacher `teacher-officialalign-low37000-20260607-2116`：tmux `vbc_teacher_g1` 仍在运行，进程 `python train_multistate.py ...` 正常存在；GPU1 约 `10263MiB/24576MiB`、利用率 `91%`。训练进度约 `36376/60000`，最新 checkpoint 为 `agent_36000.pt`，`best_agent.pt` 时间戳为 `2026-06-08 09:50`。严格解析日志得到 `Total success rate` 最新约 `0.04365`，严格峰值约 `0.04607`。当前累计成功率已略高于上一轮 `teacher-cubefallfix-low37000-20260604-1530` 训练末尾约 `0.0385`，但仍远低于官方 W&B 成功 run，需等训练完成后做 headless/termination probe 才能判断是否真正更好。
- 2026-06-08 20:45 CST 复查确认新 teacher 已完成：远端无 `vbc_teacher_g1` tmux、无 `train_multistate.py` 训练进程；GPU0/GPU1 基本空闲。日志显示 `Teacher training python exit code: 0`，并于 `2026-06-08 18:38:35 CST +0800` 正常退出。进度达到 `60000/60000`，最终 checkpoint `agent_60000.pt` 已生成；严格解析 `Total success rate` 末值约 `0.03259`，峰值仍约 `0.04607`。`best_agent.pt` 时间戳为 `2026-06-08 12:58`，与 `agent_44500.pt` 对齐；后续应先用 `best_44500.pt -> best_agent.pt` 和 `agent_60000.pt` 做 termination/headless probe。
- 2026-06-08 21:08 CST 已完成新 teacher 初步独立 termination probe：`best_44500.pt -> best_agent.pt`、`agent_60000.pt`、`agent_36000.pt` 均跑 `1000` steps、34 env，三者 `new_success=0`、`window_success_rate=0.0`、`max_lifted_object_count=0`。具体为：`best_44500` 为 `0/10`，最大高度约 `0.087m`；`agent_60000` 为 `0/8`，最大高度约 `0.093m`；`agent_36000` 为 `0/20`，最大高度约 `0.083m`。该结果说明本轮官方环境对齐 teacher 没有超过上一轮候选 `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt` 的 1000-step probe `7/23` 成功。
- 2026-06-08 21:46 CST 诊断更新：reset 分布采样确认官方对齐环境的桌面高度实际覆盖约 `0.00-0.60m`，均值约 `0.30m`；`cube_z - table_height - init_height` 接近 `0`，说明物体 reset 后贴在桌面上，不是悬空/穿桌导致 probe 失败。
- 2026-06-08 21:55 CST 固定桌高 v2 sweep 有效完成：脚本已修正为每档前强制全量 reset，`observed_table_heights` 的 min/max 与固定值一致。结果显示上一轮候选 `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt` 在固定桌高 `0.1-0.6m` 下均有成功信号，明显强于最新 `teacher-officialalign-low37000-20260607-2116/best_44500.pt`。因此“最新 official-align 独立 probe 失败”不能简单归因于 probe 脚本或高桌分布；当前更像该轮 teacher 本身学弱了。
- 2026-06-08 23:53 CST 已启动 table/reset 短 ablation：本地和远端 `high-level/envs/b1z1_pickmulti.py` 只恢复上一轮较强的 table/reset 逻辑，保留官方 `cube_falls` 判定和 `low_policy_path: "data/low_policy/publiccheckrollrew_37000.pt"`。本地 `python3 -m py_compile` 通过；远端 `python3 -m py_compile` 通过；远端低层 `publiccheckrollrew_37000.pt` 仍存在。
- 本轮没有使用带 `--delete` 的全量 `rsync`，因为仓库本地没有 `high-level/data/low_policy/publiccheckrollrew_37000.pt`，全量同步可能误删远端已有低层权重；本轮只用 `scp` 精确同步了 `high-level/envs/b1z1_pickmulti.py`。
- table/reset ablation full-scale smoke 已通过：`TEACHER_SMOKE_ENVS=10240`、`TEACHER_SMOKE_TIMESTEPS=24`、GPU1、退出码 `0`，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260608-234017.log`。
- table/reset ablation 短训已启动：run `teacher-tablereset-ablation-low37000-20260608-2341`，tmux `vbc_teacher_ablate_g1`，GPU1，`TEACHER_TIMESTEPS=10000`，日志 `/home/ubuntu/vbc-remote/remote-logs/teacher-tablereset-ablation-low37000-20260608-2341.teacher.log`。2026-06-08 23:51 CST 首个 checkpoint `agent_500.pt` 已生成，大小 `20622586` bytes；同目录 `best_agent.pt` 也已生成。500 步附近最新严格 `Total success rate` 约 `6.7e-05`，只说明出现极早期微弱成功信号，不能据此判断收敛。
- 2026-06-09 11:05 CST SSH 连接已从 `/tmp/vbc_remote_ed25519` 临时 key 恢复为稳定本地 key `/home/hjr/projects/2-Nexus/VBC/.secrets/ssh/vbc_remote_ed25519`；已验证远端返回 `SSH_OK`。私钥不进入 Git 仓库，私有 `remote-run/local/remote.env` 和外层 `agent-log/SSH.md` 已更新。
- 2026-06-09 02:59 CST table/reset ablation 短训正常完成：日志包含 `Teacher training python exit code: 0` 和 `Teacher training exited at 2026-06-09 02:59:51 CST +0800`；训练进度 `10000/10000`；最终严格 `Total success rate≈0.0195169`。最终 checkpoint `agent_10000.pt` 已生成，`best_agent.pt` 时间戳与 `agent_3000.pt` 对齐。
- 原自动 watcher `vbc_tablereset_postprobe` 因训练 tmux 会话残留误判训练仍在运行，已停止；已用手动 tmux `vbc_tablereset_manual_probe` 补跑独立 probe。
- table/reset ablation 独立 probe 结果：
  - `best_10000.pt -> best_agent.pt`，1000 steps、34 env：`new_success=1`、`new_episodes=10`、`window_success_rate=0.1`、`max_lifted_object_count=1`、`max_curr_height≈0.373m`。
  - `agent_10000.pt`，1000 steps、34 env：`new_success=10`、`new_episodes=61`、`window_success_rate≈0.1639`、`max_lifted_object_count=1`、`max_curr_height≈0.407m`。
  - 对比：弱于当前最强 `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt` 的 1000-step probe `7/23≈0.3043`，但强于 latest official-align 的 `0/10`。

远端策略：

- 双 RTX 3090 足够支撑 teacher 训练，但本项目脚本默认不是多卡训练；两张卡不会自动合并成 48GB 显存。
- 默认先在 GPU0 上训练 teacher，GPU1 保留给评估、第二 seed 或后续 student。
- teacher 默认使用 `numEnvs=10240`、`timesteps=60000`。
- 先跑 `TEACHER_SMOKE_ENVS=10240`、`TEACHER_SMOKE_TIMESTEPS=24` 的短 smoke test，确认远端不 OOM，再启动长训。
- 默认禁用 W&B，避免远端首次启动时卡在登录；如需 W&B，远端设置 `VBC_USE_WANDB=1` 和相关账号环境变量。

远端现状：

- 远端主机：`ubuntu@<remote-host>:22`
- 远端工作目录：`/home/ubuntu/vbc-remote`
- 远端环境：`vbc-py38`
- teacher smoke 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260530-161843.log`
- teacher 长训日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`
- teacher checkpoint 目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints`
- checkpoint 数量：常规 `agent_*.pt` 共 `120` 个，目录总占用约 `2.4G`。
- 关键权重：`agent_60000.pt`、`best_agent.pt`，单个文件约 `20M`。
- 本地同步目录：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-20260530-162617/`
- 本地已同步权重：`agent_60000.pt`、`agent_58000.pt`、`best_agent.pt`。
- 本地已同步评估日志：`teacher-eval-agent60000-20260531-131714.log`、`teacher-eval-agent58000-20260531-132711.log`、`teacher-eval-best-agent-20260531-132628.log`。
- 低层重训 run：`publiccheckrollrew-retrain-20260531-2354`，已完成到 `model_45000.pt`，log `/home/ubuntu/vbc-remote/remote-logs/publiccheckrollrew-retrain-20260531-2354.low.log`。
- 低层 `model_42000.pt` 已准备为高层低层策略：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_42000.pt`。
- 低层 `model_42000.pt` 对应 teacher tmux：已退出，无 tmux server。
- 低层 `model_42000.pt` 对应 teacher log：`/home/ubuntu/vbc-remote/remote-logs/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000.teacher.log`。
- 低层 `model_42000.pt` 对应 teacher checkpoint 目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/checkpoints`，最终常规 checkpoint 为 `agent_60000.pt`，另有 `best_agent.pt`，但评估未通过。
- 当前修复后 teacher tmux：已退出，训练完成。
- 当前修复后 teacher run：`teacher-cubefallfix-low37000-20260604-1530`。
- 当前修复后 teacher log：`/home/ubuntu/vbc-remote/remote-logs/teacher-cubefallfix-low37000-20260604-1530.teacher.log`。
- 当前修复后 teacher run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530`。
- 当前修复后 teacher 候选本地结果目录：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/`。
- 当前修复后低层策略：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_37000.pt`。
- 当前 student tmux：已退出；2026-06-07 13:57 CST 确认训练完成且无训练进程。
- 当前 student run：`student-best53000-display0-20260606-1145`。
- 当前 student log：`/home/ubuntu/vbc-remote/remote-logs/student-best53000-display0-20260606-1145.student.log`。
- 当前 student run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145`。
- 当前 student 最新进度：首轮旧日志停在 `10159/60000` 后从 `agent_10000.pt` 续训；续训段已完成 `50000/50000`，等效总进度 `60000/60000`。最新常规 checkpoint 为 `agent_60000.pt`，另有 `best_agent.pt`。
- 当前 table/reset ablation teacher tmux：已退出；训练完成。
- 当前 table/reset ablation teacher run：`teacher-tablereset-ablation-low37000-20260608-2341`。
- 当前 table/reset ablation teacher log：`/home/ubuntu/vbc-remote/remote-logs/teacher-tablereset-ablation-low37000-20260608-2341.teacher.log`。
- 当前 table/reset ablation teacher run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-tablereset-ablation-low37000-20260608-2341`。
- 当前 table/reset ablation 最新已确认 checkpoint：`agent_10000.pt`；`best_agent.pt` 对齐 `agent_3000.pt` 附近。

最短启动流程：

1. 本机复制 `remote-run/local/remote.env.example` 为 `remote-run/local/remote.env` 并填写远端信息。
2. 本机运行 `bash remote-run/local/sync_to_remote.sh`。
3. 远端进入 `REMOTE_ROOT` 后依次运行：
   - `bash remote-run/remote/00_probe_remote.sh`
   - `bash remote-run/remote/10_install_env.sh`
   - `bash remote-run/remote/20_prepare_project.sh`
   - `bash remote-run/remote/30_check_imports.sh`
   - `bash remote-run/remote/40_smoke_teacher.sh`
   - `bash remote-run/remote/50_start_teacher_tmux.sh`

当前注意事项：

- 当前项目工作树还有多个已修改文件，远端同步会按当前工作树状态同步，不会回滚用户或先前会话改动。
- `remote-run/` 不保存 SSH 密码；建议使用临时 SSH key。
- 如果远端不能访问外网，`10_install_env.sh` 的 PyTorch/依赖/Miniconda 安装会失败，需要改为上传离线包或使用远端已有环境。
- 低层重训已经完成，watcher 因 `python` 命令缺失失败过一次；已通过修复 `94_prepare_teacher_from_low.sh` 并手动补跑准备步骤解决。
- 低层 `model_42000.pt` 对应 teacher 已完成但评估未通过。
- 当前修复后 teacher 已完成，远端 GPU 空闲。
- 旧 teacher eval 未通过；修复后 `best_agent.pt` 是候选可用 teacher，但成功率仍低于官方成功 run，不应不经确认直接投入长时间 student 训练。

建议下一步：

1. 暂不扩展 `teacher-tablereset-ablation-low37000-20260608-2341` 到 `60000` 步；短训 probe 已表明它仍弱于 cubefallfix best。
2. 若要继续诊断 teacher，优先围绕 `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt` 做更长独立评估/视频证据，或做多 seed/依赖栈对齐，而不是继续单 seed table/reset 长训。
3. 暂不建议用 `teacher-officialalign-low37000-20260607-2116` 或 table/reset ablation 替换当前最强 cubefallfix teacher 重训 student。
4. 若要继续远端操作，使用稳定 key `/home/hjr/projects/2-Nexus/VBC/.secrets/ssh/vbc_remote_ed25519`；不要再依赖 `/tmp/vbc_remote_ed25519`。
