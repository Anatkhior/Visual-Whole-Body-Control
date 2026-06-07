# 远端快速训练启动包

用途：在拿到远端 3090/4090 机器 SSH 访问后，尽快完成 VBC 远端环境配置、smoke test，并启动 teacher/student 训练。该目录已经按当前复现实验整理进仓库，方便在其他机器 `git clone` 后继续。

## 目录

- `local/`：在本机运行，用于生成 SSH key、同步项目到远端。
- `remote/`：同步到远端后运行，用于体检、安装、验证、启动训练、监控和回传。

## 最短流程

1. 在本机准备 SSH 访问。
2. 复制 `local/remote.env.example` 为 `local/remote.env`，填写远端地址、用户、端口和远端工作目录。
3. 如需同步外部低层权重，在 `remote.env` 中填写 `LOW_POLICY_LOCAL_PATH`。仓库不会提交 `.pt` 权重。
3. 执行 `bash remote-run/local/sync_to_remote.sh`。
4. SSH 到远端，进入远端工作目录。
5. 执行：

```bash
bash remote-run/remote/00_probe_remote.sh
bash remote-run/remote/10_install_env.sh
bash remote-run/remote/20_prepare_project.sh
bash remote-run/remote/30_check_imports.sh
bash remote-run/remote/40_smoke_teacher.sh
bash remote-run/remote/50_start_teacher_tmux.sh
```

如果要继续当前“官方环境/teacher 对齐后重训”路线，推荐先在远端准备 `publiccheckrollrew_37000.pt`，再运行：

```bash
LOW_POLICY_SRC=/home/ubuntu/vbc-remote/publiccheckrollrew_37000.pt \
LOW_POLICY_TARGET_NAME=publiccheckrollrew_37000.pt \
bash remote-run/remote/20_prepare_project.sh
```

如果只使用 Google Drive 的 `model_38000.pt` 做低层验证，可以沿用默认 `LOW_POLICY_SRC=/home/ubuntu/vbc-remote/model_38000.pt`。

## 策略

- teacher 先跑 GPU0，默认 `numEnvs=10240`，`timesteps=60000`。
- 默认不启用 W&B，避免登录阻塞；如需 W&B，在远端导出 `VBC_USE_WANDB=1` 和相关环境变量。
- 两张 3090 不会自动合并显存；默认先用单卡训练 teacher，另一张卡留作评估或第二 seed。
- student 训练依赖 teacher checkpoint，teacher 有可用 checkpoint 后再启动。
- 若远端没有 conda，`10_install_env.sh` 默认会把 Miniconda 安装到 `~/miniconda3`。如果远端不能访问外网，需要提前提供 conda 或 wheel/cache。

## 安全

脚本不保存 SSH 密码。推荐使用临时 SSH key，训练结束后从远端 `~/.ssh/authorized_keys` 删除对应公钥。
