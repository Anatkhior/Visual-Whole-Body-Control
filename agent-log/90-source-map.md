# 来源映射

本日志是选择性记录，不是完整终端逐字日志。它覆盖本次同步过程中的关键命令、错误类别、恢复路径、最终提交和验证结果。若需要完整终端输出，应回看当前会话记录。

摘要来源：

- `10-project-overview.md` 来源于 `Visual-Whole-Body-Control/README.md`、`Visual-Whole-Body-Control/high-level/README.md`、`Visual-Whole-Body-Control/low-level/README.md` 的摘要。
- `20-runtime-and-integration.md` 来源于本次执行的克隆、下载、解压、Git 修复命令，以及顶层 README 的环境安装说明。
- `40-datasets-and-artifacts.md` 来源于本次生成或保留的本地路径与 README 中提到的外部权重链接。
- `50-implementation-or-training.md` 来源于高层和低层 README 的训练入口摘要。
- 2026-05-25 用户关于“已有 `model_38000.pt` 后是否能先做仿真效果”的追问与结论，记录在 `50-implementation-or-training.md` 和 `raw-by-topic/50-implementation-or-training.raw.md`。
- 2026-05-25 用户关于“本机配置能否支持到高低层联动的最终仿真效果”的澄清与结论，记录在 `50-implementation-or-training.md`、`60-evaluation-and-validation.md` 和 `raw-by-topic/50-implementation-or-training.raw.md`。
- 2026-05-26 仿真运行修复，包括依赖安装、NumPy 降级、运行路径约束、低层回放 smoke test、高层配置切换与 checkpoint 读取验证，记录在 `20-runtime-and-integration.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md` 及对应 raw 文件。
- 2026-05-26 较长低层 headless 回放与 `--record_video` 失败诊断，记录在 `60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。其中 10:28:28 的 headless 数值回放为本轮重新运行得到的最新验证证据。
- 2026-05-26 headless 视频录制修复与 `.mp4` 验证，记录在 `60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。视频证据路径为 `low-level/logs/videos/google_drive/google_drive-0-38000.mp4`。
- 2026-05-26 高层 teacher smoke test，包括导入/config 检查、`train_multistate.py --debug --timesteps 8` 标准入口短 rollout、`--timesteps 24` PPO update OOM、`numEnvs=4` 最小环境 reset/step 验证，以及 `B1Z1PickMulti._reset_envs()` 小规模索引修复，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-05-29 远端双 3090 快速部署与训练启动准备，记录在 `01-current-state.md`、`20-runtime-and-integration.md`、`50-implementation-or-training.md`、`80-artifacts-index.md`、`raw-by-topic/20-runtime-and-integration.raw.md`、`raw-by-topic/50-implementation-or-training.raw.md`。
- 2026-05-30 远端 teacher smoke test 成功与长训启动，记录在 `01-current-state.md`、`20-runtime-and-integration.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/20-runtime-and-integration.raw.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-05-31 远端 teacher 训练完成与 `agent_60000.pt` 生成，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`。
- 2026-05-31 teacher headless eval 未通过、本地同步关键权重、暂不启动 student，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-05-31 进一步 teacher 探针诊断，包括真正 `best_agent.pt` 的 symlink 评估、`agent_60000.pt` 原 eval 起点和训练起点对照、TensorBoard 指标摘要、本地/远端 checkpoint 哈希一致性、远端 GPU 空闲确认，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-05-31 `publiccheckrollrew_42000.pt` 可获得性复查，包括本地/远端文件搜索、原始配置路径、W&B run/file/artifact/API 查询、完整 `output.log` 检索和 README 外部权重链接，记录在 `01-current-state.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-05-31 按作者低层环境重训 low-level，包括新增远端脚本、安装 `vbc-low-cu113`、低层 smoke test、W&B 禁用修复、低层长训启动和 `model_42000.pt` 后自动 teacher watcher，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`。
- 2026-06-04 新低层重训后 teacher 完训与失败原因诊断，包括远端空闲确认、run 内 `publiccheckrollrew_42000.pt` 配置核对、严格 `Total success rate` 过滤、`agent_60000.pt` 300-step 定向 probe、gripper 动作统计和“接近但不抬升”判断，记录在 `01-current-state.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-06-04 README 官方命令核对、W&B 高层公开 run 摘要对比和 termination probe，包括 W&B GraphQL 公开 API、TensorBoard event 标量、`cube_falls` reset 原因统计，记录在 `60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`。
- 2026-06-04 官方 `cube_falls` 判定修复、远端切换到 `publiccheckrollrew_37000.pt`、termination probe 口径对照、full-scale smoke pass、新 teacher `teacher-cubefallfix-low37000-20260604-1530` 启动、`agent_500.pt` 早期里程碑与监控，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-06-08 official-align teacher 完训后诊断，包括 reset 分布采样、作废的 table sweep v1、有效的 table sweep v2、`officialalign best_44500.pt` 与 `cubefallfix best_53000.pt` 固定桌高对比，记录在 `01-current-state.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`。
- 2026-06-09 稳定 SSH key 恢复、table/reset ablation 完训、自动 watcher 误等待诊断、手动 probe `best_10000.pt`/`agent_10000.pt` 结果和“不要扩展该 ablation 长训”的判断，记录在 `01-current-state.md`、`20-runtime-and-integration.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-06-05 修复后 teacher 完训、`agent_60000.pt` 与 `best_agent.pt` probe、`best_agent.pt` 文件名 bug 复现与 symlink 绕过、本地同步候选权重和 SHA256 校验，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-06-05 用户要求新建 `teacher_success_log.md`，该文件是基于上述摘要层与 raw 层记录整理出的完整 teacher 实验日志；它是面向人类阅读的阶段性汇总，不替代 `agent-log/raw-by-topic/` 中的原始证据路由。
- 2026-06-07 student 完整训练后的独立 headless 限时评估、`--record_video` 依赖失败、日志本地同步、checkpoint 大文件同步受限与远端 SHA256 记录，记录在 `01-current-state.md`、`60-evaluation-and-validation.md`、`70-diagnostics-and-dead-ends.md`、`80-artifacts-index.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`、`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`。
- 2026-06-07 官方环境/teacher 对齐后重训，包括 W&B 成功 run 保存文件复查、`b1z1_pickmulti.py` reset/table/object 关键逻辑对齐、`publiccheckrollrew_37000.pt` 路径切换、teacher 启动脚本 exit code 记录、full-scale smoke pass、新 teacher `teacher-officialalign-low37000-20260607-2116` 启动和 `agent_500.pt` 早期里程碑，记录在 `01-current-state.md`、`50-implementation-or-training.md`、`60-evaluation-and-validation.md`、`80-artifacts-index.md`、`raw-by-topic/50-implementation-or-training.raw.md`、`raw-by-topic/60-evaluation-and-validation.raw.md`。
- 2026-06-07 将续作资料整理进 Git 仓库，包括 `agent-log/`、`remote-run/`、`teacher_success_log.md`、`REPRODUCTION_HANDOFF.md`、敏感连接信息脱敏和私有配置忽略规则，记录在外层与仓库内 `agent-log/01-current-state.md`，具体文件以 `Visual-Whole-Body-Control` 的 Git 状态为准。
- 2026-06-07 仓库交接资料二次整理，包括 `REPRODUCTION_HANDOFF.md` 中文化、`remote-run/local/sync_to_remote.sh` 对外部低层权重的可选上传、`remote-run/remote/20_prepare_project.sh` 对 `LOW_POLICY_TARGET_NAME` 的支持，以及 `remote-run/README.md` 的新机器续跑说明，记录在仓库内 `agent-log/01-current-state.md` 和 `80-artifacts-index.md`。
- 2026-06-07 本地 Git 提交和推送尝试：提交位于分支 `repro-handoff-and-training-fixes`；向 `origin` 推送被 HTTPS 凭据问题阻塞。该结论来自本轮 `git commit` 和 `git push -u origin repro-handoff-and-training-fixes` 输出。
- 2026-06-08 GitHub SSH 推送成功：用户将临时 SSH 公钥加入 GitHub 后，本地使用 `/tmp/vbc_github_push_ed25519` 通过 SSH 推送分支 `repro-handoff-and-training-fixes` 到 `Anatkhior/Visual-Whole-Body-Control`。该结论来自 `git push -u git@github.com:Anatkhior/Visual-Whole-Body-Control.git repro-handoff-and-training-fixes` 输出。
- `60-evaluation-and-validation.md` 来源于本次会话中实际运行的验证命令输出，以及 2026-05-25 对 `model_38000.pt`、conda 环境、PyTorch CUDA、Isaac Gym、`skrl`、`rsl_rl/legged_gym` 导入路径的只读检查。
- `70-diagnostics-and-dead-ends.md` 来源于本次会话中失败的 `git clone`、部分克隆、`git read-tree` 和错误路径 `git hash-object` 尝试。

原始细节路由：

- 克隆、下载、解压、Git 修复细节：`raw-by-topic/20-runtime-and-integration.raw.md`
- 复现环境检查细节：`raw-by-topic/20-runtime-and-integration.raw.md`
- 训练/回放阶段判断与低层权重状态：`raw-by-topic/50-implementation-or-training.raw.md`
- 验证命令和结果：`raw-by-topic/60-evaluation-and-validation.raw.md`
- 失败路径和诊断：`raw-by-topic/70-diagnostics-and-dead-ends.raw.md`

语言约束来源：用户明确要求使用 `hybrid-project-log` skill 进行同步记录，并约束记录以中文为主，除非文件名、命令、变量等本身就是英文。
