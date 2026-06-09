# 诊断与失败路径

本次同步遇到的主要问题是 GitHub 传输不稳定。

失败路径：

- 普通 `git clone` 失败，报错包括 `Could not resolve host: github.com` 和后续 `curl 56 GnuTLS recv error (-9)`。
- 浅克隆 `--depth 1 --single-branch` 仍然出现 TLS/RPC 断包。
- 部分克隆 `--filter=blob:none` 虽然建立了 `.git` 和部分工作树，但 checkout 过程中出现缺失对象，留下 `.git/index.lock` 且没有 `.git/index`。
- 初次尝试用 `git hash-object` 补写对象时路径使用错误，和 `git -C Visual-Whole-Body-Control` 叠加后导致 Git 在 `Visual-Whole-Body-Control/Visual-Whole-Body-Control/...` 下找文件，产生大量“没有那个文件或目录”错误。

有效恢复路径：

1. 下载当前提交 `869104c31953718f30ad20675e5291fcb5c5ea23` 的 GitHub 源码包。
2. `tar -tzf` 完整遍历压缩包确认可读。
3. 解压到 `Visual-Whole-Body-Control/`，保留 `.git`。
4. 移走陈旧的 `.git/index.lock` 到 `/tmp/Visual-Whole-Body-Control.index.lock.stale`。
5. 使用仓库内相对路径执行 `git hash-object -w -- "$path"` 补写本地 blob 对象。
6. 执行 `git read-tree HEAD` 重建 `.git/index`。
7. 用 `git status --short` 和 `git fsck --no-dangling` 验证。

后续不要重复错误路径：不要在 `git -C Visual-Whole-Body-Control` 的同时把 `Visual-Whole-Body-Control/$path` 作为待哈希路径传给 `git hash-object`。

2026-05-26 仿真修复中遇到并处理的问题：

- 沙箱内缺少 `/dev/nvidia*`，导致 PyTorch 误判 CUDA 不可用；沙箱外真实运行环境可用 GPU。
- Isaac Gym 与 NumPy `1.24.4` 不兼容，报 `np.float` 不存在；已通过降级到 NumPy `1.23.5` 修复。
- 外部 `/home/hjr/Isaacgym/isaacgym/python` 的 `torch_utils.py` 缺少 `euler_from_quat`，导致低层初始化失败；改为优先使用本项目 `third_party/isaacgym/python`。
- 直接调用 conda 环境 Python 时若不设置 `LD_LIBRARY_PATH`，Isaac Gym 绑定会找不到 `libpython3.8.so.1.0`；若不设置 conda `bin` 到 `PATH`，会报 `Ninja is required to load C++ extensions`。
- high-level 验证中若先导入 `skrl` 或 `torch` 再导入 `isaacgym`，会触发 Isaac Gym 的导入顺序错误；正确顺序是先导入 `isaacgym`。
- 低层 `--record_video` 原始失败路径：headless 环境下曾出现 `SetCameraLocation: Error: could not find camera with handle -1`，ffmpeg 随后报 `Unable to parse option value "0x0" as image size`。根因是 `headless=True` 时 `graphics_device_id=-1`，camera sensor 无法可靠创建。
- 该视频路径已在 2026-05-26 修复：`--record_video` 时即使 headless 也保留非负 `graphics_device_id`，并在 camera handle、frame shape、空帧处显式报错。另加入 `fetch_results()`、固定拉近录制相机和项目内绝对视频输出路径。
- 中间失败教训：第一次修复后能生成 MP4，但跟随相机/未充分刷新导致帧变化过小；最终版本使用固定拉近相机，视频内容验证通过。

2026-05-26 高层 teacher smoke test 诊断：

- 沙箱内高层导入/config 检查失败于 `/home/hjr/.cache/torch_extensions/.../gymtorch/lock` 只读文件系统，且沙箱内 `torch.cuda.device_count()` 为 `0`；高层 Isaac Gym 检查需要沙箱外真实 GPU 环境。
- 临时高层最小脚本第一次失败于导入顺序：先导入 `torch` 再导入 `isaacgym` 会触发 `ImportError: PyTorch was imported before isaacgym modules`；重排为先 `import isaacgym` 后通过该阶段。
- `numEnvs=4` 的最小脚本第二次失败于 `B1Z1PickMulti._reset_envs()`：原分类统计逻辑在 `num_group = self.num_envs // 33` 为 `0` 时生成非整数空 tensor，索引 `success_counter` 报 `IndexError: tensors used as indices must be long, int, byte or bool tensors`。
- 已修复该小规模 smoke test 问题：分类索引现在按对象列表长度生成、过滤越界索引，并固定为 `torch.long`。
- 标准 `train_multistate.py --debug --timesteps 24` 失败不是导入、资产、低层权重或 reset 失败；它已经完成环境创建、低层策略加载和 rollout，失败点是 PPO update 的 CUDA OOM。该路径在 GTX 1660 SUPER 6GB 上不适合作为 teacher 训练验证。

2026-05-30 远端 smoke 观察：

- `TEACHER_SMOKE_ENVS=10240` 的前几十秒没有明显新输出，但这不是卡死；进程仍在，日志文件也在增长，后面会一次性刷出 rollout 进度。
- 这类大规模 Isaac Gym smoke 不要因为短时间静默就中断，先确认 `pgrep`、日志大小和 `nvidia-smi` 再判断。

2026-05-31 teacher eval 诊断：

- `play_multistate.py --checkpoint best_agent.pt` 当前不可直接运行，原因是脚本假定 checkpoint 文件名最后一段可解析为整数步数。
- `agent_60000.pt` 与 `agent_58000.pt` 在 headless eval 中均未出现成功拾取；在没有进一步诊断前，不建议启动 student。
- 完整 teacher 训练日志约 `32M`，远端到本地同步非常慢；已停止完整日志同步，只保留本地 `.partial` 和完整短评估日志。
- 后续确认 `best_agent.pt` 与 `agent_58000.pt` 哈希不同，不能把 `agent_58000.pt` 当作真正 best 的等价替代。
- 通过 `/tmp/best_58000.pt -> best_agent.pt` 临时 symlink 绕过命名 bug 后，真正 `best_agent.pt` 评估仍为 `new_success=0`；400 steps 最大抬升约 `0.109m`，低于 `0.35m` 成功阈值。
- `agent_60000.pt` 使用训练起点 `x=-2.0` 对照后仍 `new_success=0`；失败不只是 eval 起点 `(-0.85, 0, 0.55)` 导致。
- TensorBoard reward 上升但 success 不成立，当前更像“学到部分接近/奖励塑形行为，但没有学到可靠抓取抬升”，而不是 eval 脚本单点误报。
- 当前失败 teacher 不应作为 student DAgger 专家；除非用户明确接受风险，不要启动 `train_multi_bc_deter.py`。

2026-05-31 18:58:00 CST +0800 teacher 未收敛复查诊断：

- 远端没有新的训练/评估进程，问题不是“还没训练完”；teacher 已完整跑到 `60000/60000`。
- `best_agent.pt` 是按 skrl 训练过程中的 checkpoint 逻辑保存的“best”，但当前证据显示它并不是按实际 pick-up success 选出来的可用专家；真正 best 的 probe 仍 `new_success=0`。
- 训练 reward 和 success 明显脱钩：TensorBoard reward 均值上升，但日志 `Total success rate` 末值只有约 `0.252%`，峰值也只有约 `0.402%`。
- 高层 reward/curriculum 代码是分阶段的：`pick_up` 在 `global_step_counter < 20000` 或 eval 时成功即 reset，之后要求 hold；`standpick`、`command_reward`、`command_penalty` 在 `30000` 步后才生效；前向速度命令 curriculum 到 `45000` 步后才收窄。当前 run 通过了这些阶段但仍未学会可靠抓取。
- 当前最需要验证的可疑点是低层策略等价性，而不是 checkpoint tensor 形状。`model_38000.pt` 可以加载且结构匹配，但仓库原始 high-level 配置的低层路径是 `publiccheckrollrew_42000.pt`；如果这两个权重行为不同，teacher 训练会在“可运行但难以学会抓取”的状态下失败。
- 不建议在没有可用 teacher 的情况下启动 student；这不是保守等待，而是避免把失败动作作为 DAgger 监督信号。

2026-06-04 12:42:31 CST +0800 新低层重训后 teacher 失败原因复查：

- 已排除“训练还没完成”：远端没有训练/评估进程，GPU 基本空闲；新 teacher 已在 `2026-06-04 05:24:26 CST +0800` 正常退出。
- 已排除“低层路径没有切换”：新 teacher run 配置实际使用 `data/low_policy/publiccheckrollrew_42000.pt`。
- 已排除“日志解析偶然值”：严格过滤 `^Total success rate ` 后，最终 `0.00582955`、峰值 `0.01045189`，训练期最高累计成功率也只有约 `1.05%`。
- 已进一步确认“接近但不抬升”：`agent_60000.pt` 300-step probe 中平均 base-object 距离从 `0.6223m` 降到 `0.4295m`，最小末端-物体距离为 `0.0730m`，但最大物体抬升只有 `0.1084m`，没有任何 `lifted_now` 或 `lifted_object`。
- 已确认“并非完全不会闭爪”：probe 的 gripper 动作维度后期均值转负；代码中 `actions[:, 6] < 0` 会触发闭爪和 `stop_pick` 停止 base command。问题更像闭爪/末端位姿/抬升时序不稳定，而不是完全没有夹爪动作。
- `best_agent.pt` 仍不应被理解为 task-success best：skrl 的 best 保存依据是 `Reward / Total reward (mean)`，不是 `Total success rate`。
- 当前高置信度结论：现有 teacher 不是可用专家；启动 student 会把失败抓取轨迹蒸馏进去。
- 当前中等置信度可疑方向：高层 reward/curriculum 允许 reward 上升但没有足够推动成功抬升；另一个方向是重训低层虽然结构匹配且可用，但在高层闭爪/抬升阶段的行为仍可能与作者隐藏低层 checkpoint 不等价。

2026-06-04 13:03:00 CST +0800 W&B 曲线对比后的失败机制判断：

- 本次 teacher 参数基本跟 README 高层 teacher 命令一致，除 `--wandb` 被禁用外，核心训练 flags 一致。禁用 W&B 只影响日志上传，不应导致策略成功率低。
- W&B 官方高层项目公开 run 显示：使用类似 `publiccheckrollrew_<step>` 的低层 checkpoint 时，部分 run 成功率很高，部分 run 失败或 crash。说明官方流程不是“任意低层 checkpoint + 一次 teacher 训练必然成功”，低层 checkpoint 选择和训练稳定性很关键。
- 官方成功 run 的 `Reward / Total reward (mean)` 约 `90-100`，成功率多在 `0.4-0.9`；本次新 teacher 末值只有 `8.256`、峰值 `9.513`，和成功 run 不是同一量级。
- W&B 成功 run `publiccheckrollrew_37000_2` 的 PPO 配置与当前 `train_multistate.py` 默认基本一致，且同样使用 1094 维 state + feature encoder；但依赖版本为 `torch==2.1.2`、`isaacgym==1.0rc4`、`numpy==1.24.4`，本次远端高层环境为 `torch 2.4.1+cu121`，若后续严格复刻成功 run 应对齐。
- 本次 `Episode / Total timesteps (mean)` 末值约 `21.1`，明显短于 W&B 成功 run 的约 `75-96`；结合 termination probe，短 episode 主要来自 `cube_falls`，不是成功或 timeout。
- 直接失败机制：策略会快速降低末端目标 z，并在低位闭爪/接触物体，导致物体掉到桌面以下触发 reset；这解释了为什么 reward 有一些上升、接近距离变小，但成功率仍接近 `0`。
- 当前更高优先级的可疑点：重训低层和作者用于成功 run 的低层 checkpoint 行为不等价，尤其在末端低位接触/闭爪时不能稳定托住或抬起物体；其次才是 teacher PPO seed/训练方差。

2026-06-04 15:13:00 CST W&B 保存代码对比后的修复方向：

- W&B 成功 run `publiccheckrollrew_37000_2` 保存的 `data/cfg/b1z1_pickmulti.yaml` 与当前配置的主要差异是低层路径：官方为 `/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_37000.pt`，当前本地为 `data/low_policy/model_38000.pt`，远端新 teacher 为 `data/low_policy/publiccheckrollrew_42000.pt`。
- W&B 成功 run 保存的 `envs/b1z1_pickmulti.py` 与当前代码存在一个直接相关差异：官方 `cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`，当前为 `cube_falls = z_cube < self.table_heights`。
- 这会把物体只要低于桌面顶面就判定掉落；官方成功 run 允许约 `3.5cm` 的容差。考虑到当前 termination probe 中 `cube_falls=170/172`，该差异是高置信度的优先修复点。
- 当前推荐：先恢复官方 `cube_falls` 判定，再用 `publiccheckrollrew_37000.pt` 做短验证；如果 reset 结构明显改善，再启动完整 teacher。不要在当前更严格 reset 判定下继续 teacher/student。

2026-05-31 23:31:55 CST +0800 `publiccheckrollrew_42000.pt` 可获得性复查：

- 用户问题：继续确认作者原配置期望的低层 checkpoint `publiccheckrollrew_42000.pt`。
- 本地复查：`/home/hjr/projects/2-Nexus/VBC` 下没有匹配 `*publiccheck*`、`*rollrew*` 或 `*42000*.pt` 的低层权重文件。
- 原始配置依据：`git show HEAD:high-level/data/cfg/b1z1_pickmulti.yaml` 显示原始 `low_policy_path` 为 `/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt`；当前本地配置为了复现实验已改为 `data/low_policy/model_38000.pt`。
- 远端复查：`/home/ubuntu/vbc-remote` 下只找到高层 teacher 的 `agent_42000.pt`，路径为 `high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/agent_42000.pt`，文件大小约 `20.6M`；远端 `high-level/data/low_policy` 下只有 `model_38000.pt -> /home/ubuntu/vbc-remote/model_38000.pt`。
- W&B run 依据：公开项目 `ericonaldo/b1z1-low` 中 run `yf5dxfg1` 的 `displayName` 是 `publiccheckrollrew`，状态 `finished`，创建时间 `2024-06-21T09:56:18Z`。
- W&B 文件列表：该 run 公开文件只有 `artifact/*/wandb_manifest.json`、`conda-environment.yaml`、`config.yaml`、`output.log`、`requirements.txt`、`wandb-summary.json`，没有 `.pt` checkpoint。
- W&B artifact 依据：该项目级 artifact type 只有 `job`、`wandb-history`、`wandb-events`；`publiccheckrollrew` 的 output artifacts 是 history/events parquet，input job artifact manifest 只含 `requirements.frozen.txt` 和 `wandb-job.json`，没有模型文件。
- W&B 直接下载验证：`https://api.wandb.ai/files/ericonaldo/b1z1-low/yf5dxfg1/publiccheckrollrew_42000.pt` 和 `https://api.wandb.ai/files/ericonaldo/b1z1-low/yf5dxfg1/model_42000.pt` 均返回 `404`。
- W&B 完整日志验证：下载的 `output.log` 为 `149519100` bytes；`rg` 能找到 `Learning iteration 42000/45000` 和 `Learning iteration 44999/45000`，但搜不到 `publiccheckrollrew_42000`、`model_42000`、`model_*.pt`、`Saving model`、`Saved model` 或 `checkpoint`。
- 代码依据：`low-level/legged_gym/scripts/train.py` 只调用 `wandb.save(...)` 上传 `b1z1_config.py` 和 `manip_loco.py`，没有上传 checkpoint 的逻辑；`low-level/legged_gym/envs/manip_loco/b1z1_config.py` 中 `max_iterations=45000`、`save_interval=200`，说明本地训练目录理论上会周期保存模型，但当前未发现其公开上传到 W&B。
- 结论：高置信度确认 `publiccheckrollrew` 这个低层训练 run 存在并跑过 42000 iteration；中高置信度判断 `publiccheckrollrew_42000.pt` 没有在当前 repo、本机、远端或可见 W&B 文件/artifact 中公开可取。后续若要得到该精确文件，优先联系作者或确认 Google Drive 权重 `model_38000.pt` 是否就是同一 run 的附近 checkpoint。

2026-05-31 23:57:10 CST +0800 低层重训启动诊断：

- 现有远端 `vbc-py38` 环境不是作者低层环境：Python `3.8.20`，`torch 2.4.1+cu121`，`wandb 0.24.2`。因此新建低层专用环境 `vbc-low-cu113`。
- 低层首次长训 `publiccheckrollrew-retrain-20260531-2350` 失败于 W&B：`train.py` 在非 debug 模式固定 `mode="online"`，远端未登录/网络不可用时 `wandb.init` 超时并退出，错误为 `wandb.errors.CommError: Run initialization has timed out after 90.0 sec.`。
- 处理：修改 `remote-run/remote/92_start_low_level_tmux.sh`，长训命令增加 `WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true`，并重启为新 run `publiccheckrollrew-retrain-20260531-2354`。
- 重启后低层长训进入正常迭代，2026-05-31 23:56 CST 已到 `Learning iteration 33/45000`，GPU0 正常占用，说明 W&B 卡点已绕过。

2026-06-02 09:10:09 CST +0800 低层长训中断与续训：

- 复查发现远端没有 `vbc_low_g0` 或 `vbc_wait_low_teacher` tmux 会话，GPU 空闲，teacher 未启动。
- 低层日志最后停在 `Learning iteration 23545/45000`，最后 checkpoint 为 `model_23400.pt`。日志末尾没有 `Traceback`、`RuntimeError`、`CUDA out of memory`、`Killed`、磁盘满或正常退出标记。
- 远端 `uptime` 显示系统未在训练期间重启；磁盘 `/` 可用约 `146G`，内存可用约 `57G`。当前停止原因暂不明确，更像外部会话/进程中断。
- 已修改 `remote-run/remote/92_start_low_level_tmux.sh` 支持 `LOW_RESUME_RUN_NAME` 和 `LOW_RESUME_CHECKPOINT`。
- 已从 `model_23400.pt` 续训，设置 `LOW_MAX_ITERATIONS=21600`，因为 rsl_rl 的 `learn(num_learning_iterations)` 会在当前 checkpoint 基础上再跑指定数量；`23400 + 21600 = 45000`。
- 续训命令启动后，2026-06-02 09:10 CST 已到 `Learning iteration 23411/45000`，GPU0 正常占用；watcher 已重启并继续等待 `model_42000.pt`。

2026-06-04 15:29:19 CST 修复后诊断状态：

- 已执行高优先级修复：`cube_falls` 从严格 `z_cube < self.table_heights` 恢复为 W&B 成功 run 官方口径 `z_cube < (self.table_heights + 0.03 / 2 - 0.05)`。
- 旧严格口径导致的 reset 结构已被验证改善：同一旧失败 teacher 在 300-step termination probe 中，新官方口径 `cube_falls=6`，而旧严格口径统计 `cube_below_table=163`。这说明“过早判定 cube_falls”确实是之前 teacher 训练的高影响干扰项。
- 该修复并不能让旧失败 teacher 立刻成功：旧 checkpoint 仍 `new_success=0`。原因是策略已经在错误/过严 reset 口径下训练完成，行为本身没有学会稳定抬升。
- 已改用 `publiccheckrollrew_37000.pt` 启动新 teacher，原因是 W&B 成功 run `publiccheckrollrew_37000_2` 明确使用 37000 步附近低层 checkpoint；继续用 42000 步低层存在低层行为不等价风险。
- 当前新 teacher 还处于早期 `~191/60000`，成功率为 `0.0` 不足以诊断成败；后续要等至少 `agent_500.pt` 和更长曲线后再判断。

2026-06-05 13:53:03 CST 修复后 teacher 结果诊断：

- `cube_falls` 官方口径修复和 `publiccheckrollrew_37000.pt` 低层策略切换是有效方向：上一轮 teacher 末尾 `Total success rate ~0.00583`，本轮末尾提升到 `~0.03852`，且 probe 出现明确 `lifted_object` 成功。
- 但本轮仍不是官方成功 run 量级：W&B 成功 run 的多类别成功率约 `0.4-0.9`，本轮训练日志累计成功率只有约 `3.85%`，1000-step probe 的 best 窗口成功率约 `30.4%`。
- `best_agent.pt` 仍受文件名解析 bug 影响，直接评估会失败；后续评估应使用数字后缀 symlink，例如 `/tmp/best_53000.pt` 或本地 `best_53000.pt -> best_agent.pt`。
- 当前判断：`best_agent.pt` 可作为候选 teacher 继续验证或做 student smoke，但不建议未经确认直接开完整 student 长训。

2026-06-05 23:46:10 CST 关于“为何仍低于官方”的差异诊断：

- 最高优先级差异仍在 `high-level/envs/b1z1_pickmulti.py`，不是 YAML 超参。当前代码已修复 `cube_falls`，但还没有完全回到 W&B 成功 run 保存的环境逻辑。
- 仍未对齐的官方代码差异：
  - `_create_extra()` 中物体初始 z：官方为 `cube_start_pose.p.z = table_pos[-1] + obj_height`，当前为 `self.table_heights[i] + obj_height`。
  - `_reset_table()` 桌高随机范围：官方 `torch_rand_float(-0.25, 0.35, ...)`，当前 `torch_rand_float(0, 0.5, ...)`。
  - `_reset_table()` 固定桌高分支：官方 `table_heights_fix - self.table_dimz`，当前 `table_heights_fix - self.table_dimz / 2`。
  - `_reset_table()` 桌面 root z 更新：官方在 initial state 上 `+= rand_heights`，当前直接赋值为 `rand_heights - self.table_dimz / 2.0`。
  - `_reset_actors()` 调用顺序：官方先 `super()._reset_actors(env_ids)`，再 reset table/object；当前先 reset table/object，再调用 `super()`。
- 这些差异会改变桌面高度分布、物体生成高度、reset 后 table/object/base 状态的相对关系，属于会改变 MDP/物理接触分布的差异；中高置信度认为它们可以解释本轮仍低于官方成功曲线。
- 低风险/保留差异：小规模分类统计使用 `category_indices(...)` 替代 `num_group = self.num_envs // 33`，是为了支持 smoke test；`torch.range` 改 `torch.arange` 是现代化修复；空 `env_ids` guard 只避免边界崩溃。这些不太可能解释官方量级差距。
- 依赖版本仍未严格对齐：本轮 high-level 远端为 `torch 2.4.1+cu121`、`numpy 1.23.5`，W&B 成功 run 记录为 `torch==2.1.2`、`isaacgym==1.0rc4`、`numpy==1.24.4`。Isaac Gym/PhysX 接触和 CUDA kernel 行为可能受版本影响，中等置信度。
- 低层 checkpoint 仍不是作者隐藏的原始 `publiccheckrollrew_37000.pt`；本轮 `publiccheckrollrew_37000.pt` 来源是我们重训 run 的 `model_37000.pt`。即使命名和步数匹配，也不能证明低层接触/闭爪/抬升阶段行为等价；这是中高置信度风险。
- 训练随机性是次要解释：W&B 官方公开项目里也存在失败/崩溃 run，说明该任务本身不稳定；但当前仍有明确代码和依赖差异，不能先把主要原因归为 seed。

2026-06-06 11:52:44 CST student headless camera 失败路径：

- 不要直接用原始 `train_multi_bc_deter.py --headless` 启动 student。原始入口只设置 `cfg["sensor"]["enableCamera"] = True`，但 `VecTask` 还需要顶层 `enableCameraSensors=True`，否则 headless 下 `graphics_device_id=-1`，camera handle 为 `-1`。
- 失败表现 1：`AttachCameraToBody: Error: could not find camera with handle -1`，随后 `_get_camera_obs()` 中 `torch.stack(self.camera_sensor_dict["forward_seg"])` 收到 `NoneType`。
- 失败表现 2：补上 `enableCameraSensors=True` 后，如果 SSH 环境没有 `DISPLAY`，Isaac Gym camera sensors 会在初始化阶段段错误/core dump。
- 已验证根因与远端图形会话相关：Isaac Gym 自带 camera 示例不带 `DISPLAY` 会 core dump，`DISPLAY=:0` 可正常运行。
- 有效路径：student 入口设置 `enableCameraSensors=True`，并用 `STUDENT_DISPLAY=:0` 或 `DISPLAY=:0` 启动；当前完整 student 长训使用该路径。

2026-06-06 18:50:40 CST student 长训无 traceback 中断：

- 现象：`student-best53000-display0-20260606-1145` 停在 `10159/60000` 附近；远端已无 tmux 会话、无 `train_multi_bc_deter.py` 进程，GPU 空闲。
- 日志：没有 `Traceback`、`error`、`exception`、`segfault`、`Killed`、`OOM`、`CUDA out of memory`、`Student training exited` 等关键字；日志末尾只是 tqdm 进度行。
- 系统状态：远端 uptime 为 `7 days, 20:17`，说明训练期间未重启；磁盘 `/` 可用约 `118G`，不是磁盘满；`dmesg` 中只看到 11:16、11:27、11:29 的早期 camera smoke segfault，没有 14:39 附近的 student 长训 segfault/OOM 记录。
- 低置信度判断：更像外部会话/进程被终止、tmux/shell 被结束，或未被日志捕获的 native 级退出；当前没有足够证据归因到 CUDA OOM、Python 异常、磁盘或系统重启。
- 后续避免重复：续训时应从 `agent_10000.pt` 开始，并改进启动命令以记录 Python pipeline 的退出码；必要时补 `remote-run/remote/common.sh` 的 student resume 参数。

2026-06-06 20:56:29 CST student 续训脚本诊断改进：

- 已避免使用 `--resume` 自动找最新 checkpoint，因为 checkpoint 目录中存在 `best_agent.pt`；虽然当前代码已过滤 `best*`，显式 `--checkpoint` 仍更可控。
- 续训使用 `--checkpoint .../agent_10000.pt`，日志确认 `Resuming from checkpoint`，GPU 占用恢复，说明 student 权重和训练入口可从该 checkpoint 恢复。
- 新启动脚本会记录 Python exit code；如果后续再次停止但无 traceback，应优先检查新增的 `Student training python exit code` 行。

2026-06-07 20:29:53 CST student 评估与同步诊断：

- `--record_video` 评估失败原因是远端 `vbc-py38` 缺少 `imageio`：日志 `/home/ubuntu/vbc-remote/remote-logs/student-eval-agent60000-record150-20260607-1402.log` 中有 `ModuleNotFoundError: No module named 'imageio'`。这只说明视频录制依赖不完整，不说明策略本身不能执行 headless eval。
- headless 数值评估使用外层 `timeout --foreground -s INT 420s`，因此退出码 `124` 是限时截停，不是 Python traceback。两个日志尾部的 `KeyboardInterrupt` 来自 timeout 发送的中断信号。
- 评估日志中的类别字典会被 `tqdm` 回车进度条粘在同一行，不能只用 `line.startswith("{'success_rate'")` 解析；本次最终用 `text.find("{'success_rate'")` 逐段截取后 `ast.literal_eval()` 得到 220/229 条类别记录。
- student 大权重本地同步未完成：并发 `scp` 和单文件 `rsync -az --progress` 都只有约 `10-20KiB/s`，单个约 `96.5MiB` checkpoint 估算需要数小时。已终止这些慢速传输并清理不完整本地 checkpoint，避免误用半文件。
- 当前远端权重完整存在并已记录 SHA256；本地只有日志和配置完成同步。后续若必须本地推理/评估，应在网络条件更好时重新同步 `agent_60000.pt` 和/或 `best_agent.pt`。

2026-06-08 21:55 CST official-align teacher 失败诊断：

- 已排除的解释：
  - `best_agent.pt` 文件名解析 bug：本轮使用数字后缀 symlink `best_44500.pt -> best_agent.pt`，仍在随机高度 1000-step probe 中 `0/10`。
  - 低层权重未加载：reset 采样和 probe 日志均显示 `Low level pretrained policy loaded!`。
  - 物体 reset 初始悬空/穿桌：`cube_z - table_height - init_height` 均值约 `0.00015m`，物体基本贴桌。
  - 固定桌高 sweep v1：该版本未强制全体 env reset，`observed_table_heights` 混杂，已作废，不能作为结论依据。
- 当前证据：
  - official-align 环境桌面高度真实分布约 `[0, 0.6]`，比上一轮较强的 cubefallfix 训练分布 `[0, 0.5]` 更宽，且高桌物体 z 可接近 `0.72m`。
  - 修正后的固定桌高 v2 sweep 显示，在同一当前评估代码下，上一轮候选 `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt` 明显强于最新 `teacher-officialalign-low37000-20260607-2116/best_44500.pt`。
  - official-align checkpoint 在固定 `0.50/0.60m` 桌高有少量成功，但在 `0.10-0.40m` 基本失败；cubefallfix checkpoint 在 `0.10-0.60m` 均有成功信号。
- 当前判断：
  - 中高置信度：最新 official-align run 本身学弱了，不能用它继续 student 长训。
  - 中等置信度：官方 W&B 保存代码中的 table reset 分布/顺序与 PPO 随机性共同使训练更不稳定；“更接近 W&B 文件”不等于这一次 seed 会复现官方成功曲线。
  - 低到中等置信度：`_reset_actors()` 顺序中 `super()._reset_actors()` 会先触发 `update_roboinfo()`，但 reward/termination 前 `post_physics_step()` 会再次更新，因此它不是单独解释失败的强证据；更适合作为后续 ablation 项。
  - 中等置信度：依赖栈仍与 W&B 成功 run 不一致，远端是 `torch 2.4.1+cu121`，官方记录是 `torch 2.1.2`/`numpy 1.24.4`；Isaac Gym 接触数值差异可能放大训练不稳定性。
- 建议：
  - 实用路线：回退使用 `teacher-cubefallfix-low37000-20260604-1530/best_agent.pt` 作为当前最强 teacher 候选。
  - 诊断路线：保留官方 `cube_falls` 与 `publiccheckrollrew_37000.pt`，只对 table reset 范围/顺序做短 ablation 或多 seed，而不是再直接重复同一 official-align 单 seed 长训。

2026-06-09 11:43 CST table/reset ablation 后续诊断：

- `/tmp/vbc_remote_ed25519` 丢失不是远端训练问题，而是本地把 SSH 私钥放在临时目录；`/tmp` 可能被清理或随会话生命周期消失。已改用稳定本地私有路径，不再依赖 `/tmp`。
- 自动 watcher `vbc_tablereset_postprobe` 的失败原因是过度依赖 `tmux has-session -t vbc_teacher_ablate_g1`。训练 Python 已正常退出，日志也写出 exit code，但远端仍残留一个 tmux session 进程，导致 watcher 一直等待。
- 后续 watcher 不应只看 tmux session 是否存在；更可靠的终止条件是同时检查训练日志里的 `Teacher training python exit code` 或 `Teacher training exited`，再确认没有 `train_multistate.py` 进程。
- table/reset ablation 结果：短训 probe 有真实成功，但弱于 cubefallfix best；这说明“恢复上一轮 table/reset 逻辑”不是单独的充分修复项。当前更高价值方向仍是使用 cubefallfix best 作为候选，或做依赖栈/低层 checkpoint/seed 对齐。
