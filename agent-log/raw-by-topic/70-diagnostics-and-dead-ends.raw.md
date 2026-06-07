# 诊断与失败路径原始记录

<!-- source: session 2026-05-25, failed commands and recovery notes -->

失败现象：

- 沙箱网络下第一次 `git clone` 报 `Could not resolve host: github.com`。
- 授权网络后，多次 `git clone` 和浅克隆出现：
  - `RPC 失败。curl 56 GnuTLS recv error (-9): Error decoding the received TLS packet.`
  - `fetch-pack: unexpected disconnect while reading sideband packet`
  - `fatal: 过早的文件结束符（EOF）`
  - `fatal: fetch-pack：无效的 index-pack 输出`

部分克隆后状态：

- `.git` 存在。
- `git rev-parse HEAD` 可得到 `869104c31953718f30ad20675e5291fcb5c5ea23`。
- `git status --short` 曾大量显示 `D` 和 `??`，因为 `.git/index` 缺失且工作树不完整。
- `ls -la Visual-Whole-Body-Control/.git/index.lock Visual-Whole-Body-Control/.git/index` 显示 `.git/index.lock` 为 0 字节，`.git/index` 不存在。

诊断结论：

- GitHub 传输中断导致部分克隆进入不完整 checkout 状态。
- 源码包下载成功后，工作树内容可补齐，但本地 `.git` 因部分克隆缺少 blob，不能直接 `git read-tree HEAD`。
- 通过 `git hash-object -w -- "$path"` 把工作树文件写入本地对象库后，`git read-tree HEAD` 可以成功重建索引。

错误尝试：

- 曾在 `git -C Visual-Whole-Body-Control` 的同时给 `git hash-object` 传入 `Visual-Whole-Body-Control/$path`，导致路径被错误解释为仓库内的 `Visual-Whole-Body-Control/...`，产生大量“没有那个文件或目录”错误。

<!-- source: session 2026-05-26 09:09:37 CST +0800, simulation repair diagnostics -->

仿真修复期间的失败路径和结论：

- 初始 CUDA 误判：
  - 沙箱内 `torch.cuda.is_available()` 为 `False`，并有 `Can't initialize NVML`。
  - 沙箱外同一环境中 CUDA 为 `True`，因此不是驱动或 PyTorch 安装整体损坏，而是沙箱设备可见性问题。
- NumPy 版本问题：
  - `play.py` 初次启动失败于 `/home/hjr/Isaacgym/isaacgym/python/isaacgym/torch_utils.py` 的 `dtype=np.float`。
  - 根因是 NumPy `1.24.4` 移除了 `np.float`。
  - 修复为 NumPy `1.23.5` 后，`np.float` 可用。
- Isaac Gym 路径问题：
  - 使用外部 `/home/hjr/Isaacgym/isaacgym/python` 时，低层初始化报 `NameError: name 'euler_from_quat' is not defined`。
  - 本项目自带 `Visual-Whole-Body-Control/third_party/isaacgym/python/isaacgym/torch_utils.py` 包含 `euler_from_quat`。
  - 结论：VBC 运行时必须把项目内 `third_party/isaacgym/python` 放在 `PYTHONPATH` 最前面。
- 直接 Python 运行环境问题：
  - 不设置 `LD_LIBRARY_PATH=/home/hjr/miniconda3/envs/legged-nexus-py38/lib` 时，导入 Isaac Gym 绑定报 `libpython3.8.so.1.0` 找不到。
  - 不把 `/home/hjr/miniconda3/envs/legged-nexus-py38/bin` 放入 `PATH` 时，加载 `gymtorch` 报 `Ninja is required to load C++ extensions`。
- high-level 导入顺序问题：
  - 先导入 `skrl` 或 `torch` 再导入 `isaacgym`，会报 `PyTorch was imported before isaacgym modules`。
  - 正确顺序是先 `import isaacgym`，再导入其他会触发 PyTorch 的模块；项目入口脚本本身符合这个顺序。

<!-- source: session 2026-05-26 09:42:41 CST +0800, low-level video capture failure -->

低层 `--record_video` 尝试失败：

- 现象：
  - 策略与环境可启动，checkpoint 可加载。
  - headless camera 路径反复输出：`SetCameraLocation: Error: could not find camera with handle -1`。
  - ffmpeg 随后报：`Unable to parse option value "0x0" as image size`。
  - 未得到可作为验收证据的 `.mp4`。
- 判断：
  - 这是 viewer/video capture 路径问题，不是低层 checkpoint 读取失败。
  - 当前视觉证据不足；若要人工观察运动，需要修复 camera handle 或改用真实 viewer。
  - 已用同一低层 checkpoint 的 `1000` steps / `20.0` 秒 headless 数值回放补充验证运动存在。

<!-- source: session 2026-05-26 11:06:53 CST +0800, video capture repair diagnostics -->

低层 headless 视频录制修复诊断：

- 根因：
  - `BaseTask.__init__` 在 `headless=True` 时无条件设置 `graphics_device_id=-1`。
  - Isaac Gym camera sensor 录制即使不创建 viewer，也需要非负 graphics device；否则 camera handle 可能为 `-1`，后续 `set_camera_location` 和 ffmpeg 写入会失败或生成 `0x0` 帧。
- 有效修复：
  - `low-level/legged_gym/envs/base/base_task.py`：当 `cfg.env.record_video=True` 时，即使 headless 也保留 `graphics_device_id=self.sim_device_id`。
  - `low-level/legged_gym/envs/manip_loco/manip_loco.py`：camera handle 小于 `0` 时立即抛错；读取帧时校验 shape 和空帧；写入 RGB 连续数组。
  - `low-level/legged_gym/envs/manip_loco/manip_loco.py`：camera 渲染前显式 `fetch_results(self.sim, True)`；最终使用固定拉近相机 `[-2.2, 2.2, 1.3]` 和 `horizontal_fov=45.0`，让运动在画面中更明显。
  - `low-level/legged_gym/scripts/play.py`：视频输出路径改为 `LEGGED_GYM_ROOT_DIR/logs/videos/<run_name>` 的项目内绝对路径，并打印 `Recording video to: ...`。
- 中间无效/弱证据路径：
  - 只修 graphics device 后可以生成 `.mp4`，但跟随相机和刷新不足导致抽样帧差过小。
  - 加 `fetch_results()` 后视频变化增大，但远距离固定相机下全画面平均差仍偏小。
  - 拉近固定相机并缩小 FOV 后，内容验证通过。

<!-- source: session 2026-05-26 11:44:45 CST +0800, high-level teacher smoke diagnostics -->

高层 teacher smoke test 的失败路径与处理：

- 沙箱内运行 high-level import/config 检查失败：
  - 报错：`OSError: [Errno 30] Read-only file system: '/home/hjr/.cache/torch_extensions/py38_cu121/gymtorch/lock'`
  - 同时沙箱内 CUDA device count 为 `0`。
  - 处理：改为沙箱外授权运行，因为 Isaac Gym 需要真实 GPU 设备和可写 torch extension cache。
- 临时最小脚本第一次失败：
  - 报错：`ImportError: PyTorch was imported before isaacgym modules. Please import torch after isaacgym modules.`
  - 处理：把 `import isaacgym` 放在 `import torch` 前。
- 临时最小脚本第二次失败：
  - 报错：`IndexError: tensors used as indices must be long, int, byte or bool tensors`
  - 位置：`high-level/envs/b1z1_pickmulti.py` 的 `_reset_envs()`，索引 `self.success_counter[bowl_indices]`。
  - 根因：原代码使用 `num_group = self.num_envs // 33`。当 `numEnvs=4` 时 `num_group=0`，`np.array([...]).reshape(1,-1).squeeze()` 生成空浮点数组，转成 torch 后不能作为索引。
  - 修复：新增局部 `category_indices(offsets)`，按 `len(self.obj_list)` 生成索引、过滤超过 `self.num_envs` 的项，并返回 `dtype=torch.long` tensor。
- 标准 `train_multistate.py --debug --timesteps 24` 失败：
  - 失败点在 PPO update，不在环境创建或低层策略加载。
  - 报错：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB.`
  - 结论：GTX 1660 SUPER 6GB 可做短 rollout，但不适合作为 teacher PPO 训练机器。

<!-- source: session 2026-05-30 16:18:43 CST +0800, remote smoke waiting/monitoring -->

远端 smoke 的观察性诊断：

- `TEACHER_SMOKE_ENVS=10240` 的前几十秒没有明显新输出，但这不是卡死；进程仍在，日志文件也在增长，后面会一次性刷出 rollout 进度。
- 这类大规模 Isaac Gym smoke 不要因为短时间静默就中断，先确认 `pgrep`、日志大小和 `nvidia-smi` 再判断。

<!-- source: session 2026-05-31 13:17:14-16:26:39 CST +0800, teacher eval diagnostics -->

teacher eval 诊断：

- `play_multistate.py` 的 `trainer.eval()` 未按传入的 `--timesteps 3000` 停止，进度条仍显示 `50000`。后续短评估应使用外层 `timeout` 或写专门 eval 脚本。
- `best_agent.pt` 文件名不符合当前 `train_multistate.py` 的 eval 解析逻辑；要直接评估它，需要复制/链接为类似 `agent_<step>.pt` 的名字，或修改解析逻辑。

<!-- source: session 2026-05-31 17:02:46 CST +0800, teacher eval failure diagnosis -->

teacher 效果确认失败的追加诊断：

- 初步误区：`best_agent.pt` 时间戳与 `agent_58000.pt` 接近，但二者 SHA256 不同，因此不能把 `agent_58000.pt` 当作真正 best 的等价替代。
- 已用 `/tmp/best_58000.pt -> best_agent.pt` 绕过命名解析 bug 评估真正 best；结果仍 `new_success=0`。
- `agent_60000.pt` 在原 eval 起点和训练起点 `x=-2.0` 下都没有成功；失败不只是 eval 起点差异造成。
- 三次 probe 均出现末端能接近物体的迹象：例如 `best_agent.pt` 的 `min_curr_dist=0.019195`，但最大抬升只有约 `0.109m`，没有达到 `0.35m` 成功阈值。
- TensorBoard reward 上升但 success 不成立，说明 reward 改善不能直接证明策略会拾取。
- 当前不建议继续 student。风险是 student 会学习失败 teacher 的动作标签，消耗远端时间但无法得到视觉拾取策略。
- `agent_60000.pt` 和 `agent_58000.pt` 的 headless eval 均为 `Total success rate 0.0`，不建议据此启动 student。
- 同步完整 teacher 训练日志时，`teacher-20260530-162617.teacher.log` 远端文件约 `32M`，传输速度极低；已停止完整日志同步，并将本地部分文件标记为 `.partial`。

<!-- source: session 2026-05-31 18:41:27-18:58:00 CST +0800, teacher failure root-cause recheck -->

teacher 未收敛根因复查记录：

- 已排除/弱化：
  - 远端训练未完成：不成立。训练已退出，日志显示 `60000/60000`。
  - 后台还有 student 或 teacher 在跑：不成立。远端进程检查返回 `no_active_process`。
  - 本地同步权重损坏：不成立。远端与本地三份关键权重 SHA256 一致。
  - `best_agent.pt` 文件名解析导致的假失败：不成立。该 bug 存在，但通过 `/tmp/best_58000.pt` 临时链接评估真正 best 后仍 `new_success=0`。
  - 低层 checkpoint 结构不匹配：低置信度可疑。`model_38000.pt` 的关键 tensor 形状与 `high-level/utils/low_level_model.py` 对齐。
- 仍然可疑：
  - 低层 checkpoint 行为等价性。仓库原始 `high-level/data/cfg/b1z1_pickmulti.yaml` 指向 `/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt`，当前为运行改成 `data/low_policy/model_38000.pt`。结构一致不等于行为一致。
  - Reward shaping 与 success 目标脱钩。TensorBoard reward 上升，但训练日志最终 `Total success rate` 约 `0.00252265`，峰值约 `0.00401653`。
  - `best_agent.pt` 更可能是按 reward/训练内部指标保存，而不是按真实 eval success 保存，因此不能仅凭文件名判定可用。
- 代码依据：
  - `high-level/envs/reward_vec_task.py`：`_reward_pick_up()` 在 `global_step_counter < 20000 or self.eval` 时 lifted 即 reset；之后要求 `pick_counter >= hold_steps`。
  - `high-level/envs/reward_vec_task.py`：`_reward_command_reward()`、`_reward_command_penalty()` 在 `global_step_counter < 30000` 时为 `0`。
  - `high-level/envs/b1z1_pickmulti.py`：`_reward_standpick()` 在 `global_step_counter < 30000` 时为 `0`。
  - `high-level/envs/b1z1_base.py`：`clip_commands()` 的前向速度范围随 `15000/30000/45000` 步分段收窄。
- 当前决策：
  - 不启动 student。
  - 下一步优先找/确认作者 high-level 训练时使用的低层 checkpoint，或者做一个低层行为等价 probe，再考虑是否重训 teacher。

<!-- source: session 2026-05-31 23:31:55 CST +0800, publiccheckrollrew checkpoint availability check -->

`publiccheckrollrew_42000.pt` 复查原始要点：

- 用户要求：继续确认 `publiccheckrollrew_42000.pt`。
- 本地命令：`find /home/hjr/projects/2-Nexus/VBC -type f \( -iname '*publiccheck*' -o -iname '*rollrew*' -o -iname '*42000*.pt' \) -print`，无输出。
- 原始配置命令：`git -C Visual-Whole-Body-Control show HEAD:high-level/data/cfg/b1z1_pickmulti.yaml`，原始 `low_policy_path` 指向 `/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt`。
- 远端命令：`find /home/ubuntu/vbc-remote -type f \( -iname '*publiccheck*' -o -iname '*rollrew*' -o -iname '*42000*.pt' \)`，只返回高层 teacher `agent_42000.pt 20622806`。
- 远端低层目录：`high-level/data/low_policy` 只有 `model_38000.pt -> /home/ubuntu/vbc-remote/model_38000.pt`。
- W&B GraphQL run 查询返回：项目 `ericonaldo/b1z1-low`，run `yf5dxfg1`，`displayName=publiccheckrollrew`，`state=finished`，`createdAt=2024-06-21T09:56:18Z`。
- W&B run 文件列表只包括：`artifact/1195249623/wandb_manifest.json`、`artifact/2099896402/wandb_manifest.json`、`artifact/932810060/wandb_manifest.json`、`conda-environment.yaml`、`config.yaml`、`output.log`、`requirements.txt`、`wandb-summary.json`。
- W&B artifact 查询返回：output artifacts 为 `run-yf5dxfg1-history` 和 `run-yf5dxfg1-events`；input artifact 为 job artifact。manifest 内容分别是 `0000.parquet`、`sketch/sketch.parquet`、`events_0000.parquet`、`requirements.frozen.txt`、`wandb-job.json`。
- W&B 项目级 artifact type 查询返回：`job`、`wandb-history`、`wandb-events`；没有 `model` 或 checkpoint 类型。
- 直接下载 `publiccheckrollrew_42000.pt` 和 `model_42000.pt` 均为 `404`。
- 下载完整 `output.log` 到 `/tmp/vbc_publiccheckrollrew_output.log`，大小 `149519100` bytes。`rg` 结果：
  - `2730096: Learning iteration 42000/45000`
  - `2925027: Learning iteration 44999/45000`
  - 搜索 `publiccheckrollrew_42000|model_42000|model_[0-9]+\.pt|Saving model|Saved model|checkpoint` 无结果。
- `wandb-summary.json` 显示 `_step=44999`、`_runtime=100219.79723668098`、`Train/mean_reward=21.312902555465698`、`Train/mean_arm_reward=3.317793595790863`、`Perf/total_fps=69245`。
- `requirements.txt` 显示原始环境包括 `python 3.8.18`、`torch==1.10.0+cu113`、`torchvision==0.11.1+cu113`、`torchaudio==0.10.0+cu113`、`numpy==1.23.5`、`isaacgym==1.0rc3`、`rsl-rl==1.0.2`、`wandb==0.16.2`。
- 诊断结论：该 run 的确公开存在且跑过 42000 iteration；目标 checkpoint 没有出现在当前可见的公开分发面上。

<!-- source: session 2026-06-04 12:35:57-12:42:31 CST +0800, diagnosis after retrained low-level teacher probe -->

新低层重训后 teacher 失败原因诊断原始记录：

- 用户问题：先调查目前新 teacher 训练完成但效果不佳的原因。
- 已排除项：
  - 训练未完成：远端无 `train_multistate.py`、`train.py`、`vbc_teacher_probe` 后台进程，GPU 利用率 `0%`，teacher 日志有 `Teacher training exited at 2026-06-04 05:24:26 CST +0800`。
  - 低层路径没切换：run 内 YAML 已是 `low_policy_path: data/low_policy/publiccheckrollrew_42000.pt`。
  - `best_agent.pt` 文件名 bug：该 bug 存在，但之前已用 symlink 评估真正 best，仍 `new_success=0`；本轮 `agent_60000.pt` 也失败。
  - eval 起点单点问题：之前训练起点 `x=-2.0` 对照仍 `new_success=0`。
- 成功率证据：
  - 严格过滤 `^Total success rate ` 后：`strict_success_count=59917 first=0.00000000 last=0.00582955 max=0.01045189`。
  - 日志尾部类别成功率很不均衡：例如 Bowl 约 `0.1296`，Cup 约 `0.0147`，Ball 近似 `0`，总体被大量失败类别拉低。
- 行为证据：
  - `agent_60000.pt` 300-step probe：平均 base-object 距离从 `0.6223m` 到 `0.4295m`，说明有接近行为。
  - 最小末端-物体距离 `0.0730m`，说明有时末端接近物体。
  - 最大物体抬升 `0.1084m`，远低于 `liftedSuccessThreshold=0.35m`。
  - `max_lifted_now_count=0`、`max_lifted_object_count=0`，说明没有达到任务成功判据。
  - gripper 动作维度后期均值为负；结合代码 `actions[:, 6] < 0` 表示闭爪，说明不是完全没学会闭爪，而是闭爪/末端位姿/抬升时序没有形成稳定成功。
- 代码依据：
  - `high-level/envs/reward_vec_task.py`：`approaching` 和 `lifting` 都是增量塑形奖励，reward 改善可能不等同于达到最终 pick-up 成功。
  - `high-level/envs/b1z1_pickmulti.py`：`lifted_object` 需要高度超过阈值并且末端距离 `<0.1`；这是当前 probe 未达成的硬条件。
  - `high-level/envs/b1z1_base.py`：高层动作通过 `curr_ee_goal_cart` 增量、`curr_ee_goal_orn_rpy` 增量、`actions[:, 6]` 二值夹爪、`actions[:, 7:9]` base command 进入低层控制。
  - `third_party/skrl/skrl/agents/torch/base.py`：`best_agent.pt` 由 `Reward / Total reward (mean)` 选择，不按 `Total success rate` 选择。
- 当前判断：
  - 高置信度：现有 teacher 不是可用专家，不应启动 student。
  - 高置信度：失败模式是“靠近和尝试闭爪存在，但没有稳定抓取抬升”。
  - 中等置信度：reward/curriculum 与成功率脱钩是主要训练层面原因之一。
  - 中等置信度：重训低层虽然可加载且训练完成，但其高层闭爪/抬升阶段行为仍可能与作者隐藏的 `publiccheckrollrew_42000.pt` 不完全等价。
  - 低置信度：单纯延长当前 teacher 同配置训练可以解决；现有成功率峰值只有约 `1.05%`，没有明显收敛趋势。

<!-- source: session 2026-06-04 15:13:00-15:29:19 CST +0800, post-fix diagnosis -->

官方 `cube_falls` 修复后的诊断原始记录：

- 高置信度修复点已执行：`cube_falls` 恢复为 W&B 成功 run 保存代码中的官方容差口径。
- 旧失败 teacher 的 termination probe 显示：
  - 新官方口径 `cube_falls=6`。
  - 旧严格口径 `cube_below_table=163`。
  - 该差异解释了为什么旧训练中 `cube_falls` reset 过多；过严格判定会让 episode 过短，破坏抓取/抬升阶段学习。
- 旧 teacher 没有因为判定修复而立即成功：
  - `new_success=0`
  - 说明旧策略本身已经学成失败行为，不能直接作为 student 专家。
- 新 teacher 目前只处于 `~191/60000` 的很早期：
  - 现在 `Total success rate 0.0` 不构成失败证据。
  - 后续至少应等待 `agent_500.pt`、`agent_1000.pt` 以及更长 reward/success 曲线，再判断是否保留这条训练方向。

<!-- source: session 2026-06-05 12:32:27-13:53:03 CST +0800, post-completion diagnosis -->

修复后 teacher 完成后的诊断原始记录：

- `cube_falls` 修复方向被结果支持：本轮训练至少学到了部分成功拾取，`best_agent.pt` 1000-step probe 中 `new_success=7`。
- 训练日志末尾 `Total success rate ~0.03852`，仍低于官方成功 run；这说明改动解决了主要 blocker，但没有完全复刻官方表现。
- `best_agent.pt` 文件名解析 bug 仍然存在，应继续用数字后缀 symlink 评估或后续修脚本。
- 当前风险判断：
  - 用 `best_agent.pt` 做 student smoke 有依据。
  - 直接投入长时间 student 训练有风险，因为 teacher 不是强专家。

<!-- source: session 2026-06-07 14:02-20:29 CST +0800, student eval video/sync diagnostics -->

Student 评估视频与权重同步诊断原始记录摘要：

- `--record_video` 失败：
  - 日志：`/home/ubuntu/vbc-remote/remote-logs/student-eval-agent60000-record150-20260607-1402.log`
  - 本地同步日志：`remote-results/student-best53000-display0-20260606-1145/logs/student-eval-agent60000-record150-20260607-1402.log`
  - 关键错误：`ModuleNotFoundError: No module named 'imageio'`
  - 结论：这是远端环境缺少视频写入依赖，不是 headless 数值评估失败。
- 限时 headless eval 的 `KeyboardInterrupt`：
  - 两个 headless 日志尾部都有 `KeyboardInterrupt`，同时外层记录 exit code `124`。
  - 原因是 `timeout --foreground -s INT 420s` 到时发送中断信号；这是本轮为了快速诊断采用的预期截停，不是策略 traceback。
- 类别字典解析坑：
  - 日志中类别字典形如 `{'success_rate': {'SuccessRate / Bowl': ...}}`。
  - 由于 `tqdm` 使用回车刷新，字典可能粘在进度条同一行，不能可靠使用 `line.startswith(...)`。
  - 本轮最终用文本搜索 `"{'success_rate'"` 后截取到换行/回车，再 `ast.literal_eval()` 解析。
- student checkpoint 同步失败路径：
  - 并发 `rsync -az` 同步两个 checkpoint 时，`agent_60000.pt` 临时文件停在 `0 bytes`；训练日志同步可完成。
  - 并发 `scp` 同步两个 checkpoint 时，本地文件几十秒只增长到数百 KiB。
  - 单文件 `rsync -az --progress` 同步 `best_agent.pt` 时，速度约 `10-20KiB/s`，进度输出预估 `1:11:07` 到 `3:25:04` 不等。
  - 已终止慢速 `rsync`/`scp` 进程，并删除不完整的本地 `agent_60000.pt`、`best_agent.pt` 和 `.best_agent.pt.*`/`.agent_60000.pt.*` 临时文件。
  - 远端权重仍完整存在：
    - `agent_60000.pt`：`101222307` bytes，SHA256 `fd7f1e0cf0406ed41b02b3e52737a9bc340f5a786d0329b21cd5d0167ac346c7`
    - `best_agent.pt`：`101222242` bytes，SHA256 `053e21f8a94fa6acdac1b30854350ff71a3a0bc63cf6783bb2674e6e19e2d69e`
