# 复现交接说明

这个仓库包含原项目代码、当前复现实验的代码修复、`agent-log/` 交接记录，以及 `remote-run/` 远端训练脚本。目标是让后续在其他机器 `git clone` 后可以继续推进，而不依赖当前会话上下文。

## 先读哪些文件

1. `agent-log/00-start-here.md`
2. `agent-log/01-current-state.md`
3. `agent-log/80-artifacts-index.md`
4. `teacher_success_log.md`

`agent-log/` 是主要续作记录，中文为主，记录了训练决策、失败路径、远端 run 名、checkpoint 路径、评估结果和下一步计划。

## 当前续作重点

当前路线是“官方环境/teacher 对齐后重训”。关键状态：

- 已对齐 W&B 成功 run 保存代码中的 high-level reset/table/object 关键逻辑。
- 当前 high-level 配置指向 `data/low_policy/publiccheckrollrew_37000.pt`。
- 远端新 teacher run 为 `teacher-officialalign-low37000-20260607-2116`，仍需继续监控和评估。

权重文件没有纳入 Git。若在新机器复现，需要手动准备低层 policy checkpoint，并放到 `high-level/data/low_policy/` 或通过 `remote-run` 上传到远端后再链接。

## 远端训练脚本

`remote-run/` 包含远端机器准备、smoke test、teacher/student 训练、监控和结果回传脚本。

配置新机器：

```bash
cp remote-run/local/remote.env.example remote-run/local/remote.env
```

然后填写 `REMOTE_USER`、`REMOTE_HOST`、`REMOTE_PORT`、`REMOTE_ROOT`，以及可选 `SSH_KEY`、`LOW_POLICY_LOCAL_PATH`。私有 `remote.env` 已被 `.gitignore` 排除。

## 不随 Git 提交的内容

仓库不会提交以下内容：

- 生成的 `.pt` checkpoint
- 大 `.log` 日志
- `.mp4` 视频
- 本机 SSH 私钥和明文密码
- `remote-run/local/remote.env`
- `remote-results/` 中的大型同步结果

重要产物路径、SHA256 和评估结论记录在 `agent-log/80-artifacts-index.md` 与 `teacher_success_log.md` 中。
