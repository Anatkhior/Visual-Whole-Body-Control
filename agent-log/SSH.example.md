# SSH 连接信息模板

用途：给后续会话中的 agent 快速连接可用远端机器，并避免误伤正在运行的 VBC 训练任务。

不要在公开仓库中提交明文密码、私钥内容或本机私有 `remote.env`。

## 连接参数

- Host：`<remote-host>`
- User：`<remote-user>`
- Port：`22`
- 远端根目录：`<remote-root>`，例如 `/home/<remote-user>/vbc-remote`
- 项目目录：`<remote-root>/Visual-Whole-Body-Control`
- 远端日志目录：`<remote-root>/remote-logs`

推荐使用 SSH key：

```bash
ssh -p 22 -i <ssh-key-path> -o BatchMode=yes -o StrictHostKeyChecking=accept-new <remote-user>@<remote-host>
```

复制文件：

```bash
scp -P 22 -i <ssh-key-path> <local_path> <remote-user>@<remote-host>:<remote_path>
```

## 本地配置

复制 `remote-run/local/remote.env.example` 为 `remote-run/local/remote.env`，然后填写：

```bash
REMOTE_USER=<remote-user>
REMOTE_HOST=<remote-host>
REMOTE_PORT=22
REMOTE_ROOT=<remote-root>
SSH_KEY=<ssh-key-path>
RSYNC_EXTRA_ARGS=
```

`remote-run/local/remote.env` 已被 `.gitignore` 忽略。

## 检查命令

查看 tmux：

```bash
ssh -p 22 -i <ssh-key-path> -o BatchMode=yes <remote-user>@<remote-host> tmux ls
```

查看 GPU：

```bash
ssh -p 22 -i <ssh-key-path> -o BatchMode=yes <remote-user>@<remote-host> nvidia-smi
```

查看当前 teacher：

```bash
ssh -p 22 -i <ssh-key-path> -o BatchMode=yes <remote-user>@<remote-host> \
  env TEACHER_RUN_NAME=<teacher-run-name> bash <remote-root>/remote-run/remote/60_monitor_teacher.sh
```
