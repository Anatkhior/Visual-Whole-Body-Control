# VBC Teacher 成功候选权重实验日志

更新时间：2026-06-05 14:15:28 CST +0800。

本次记录请求：2026-06-05 14:15:28 CST +0800，用户要求新建 `teacher_success_log.md`，记录从最开始克隆 `Visual-Whole-Body-Control` 到当前训练出具备一定成功率的 teacher 权重为止的完整实验过程；重要命令、操作、权重路径、可视化评估路径和效果文件路径都需要写清楚。

本文档记录从克隆 `Visual-Whole-Body-Control` 项目到当前训练出具备一定成功率的 high-level teacher 候选权重为止的主要工作、关键命令、诊断依据、权重路径和评估产物。记录以中文为主，文件名、命令、变量名、路径等保留原始英文。

## 0. 当前结论

当前最重要结果：

- 修复后 teacher run：`teacher-cubefallfix-low37000-20260604-1530`
- 远端最终训练进度：`60000/60000`
- 训练日志末尾 `Total success rate ≈ 0.0385234`
- 最终权重 `agent_60000.pt` 的 300-step probe：`5/40` 成功，窗口成功率 `12.5%`
- 最强候选 `best_agent.pt` 的 1000-step probe：`7/23` 成功，窗口成功率 `30.43%`
- 候选 teacher 权重已同步回本地，SHA256 与远端一致

本轮结果已经明显不同于早期 `new_success=0` 的失败 teacher，说明 high-level teacher 已经学到部分真实拾取能力。但它仍低于 W&B 官方成功 run 的大致 `0.4-0.9` 成功率区间，因此当前应称为“候选可用 teacher”，不能直接称为论文级完全复现。

最强候选权重：

```text
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt
SHA256: adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c
```

为绕过 `best_agent.pt` 文件名解析 bug，本地还有一个数字后缀 symlink：

```text
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt -> best_agent.pt
```

## 1. 项目目录与机器

本地工作区：

```text
/home/hjr/projects/2-Nexus/VBC/
```

项目源码：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/
```

项目日志：

```text
/home/hjr/projects/2-Nexus/VBC/agent-log/
```

远端训练机：

```text
ubuntu@<remote-host>:22
远端工作目录: /home/ubuntu/vbc-remote
远端项目目录: /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/
```

远端为双 RTX 3090。项目训练脚本默认不是多卡训练，所以本次训练仍按单卡任务调度：GPU0 常用于低层或评估，GPU1 常用于 teacher 长训。

## 2. 克隆与恢复项目

最初目标是克隆官方 GitHub 项目：

```bash
git clone https://github.com/Anatkhior/Visual-Whole-Body-Control
```

普通 `git clone` 曾遇到网络与 TLS/RPC 中断问题，包括：

- `Could not resolve host: github.com`
- `curl 56 GnuTLS recv error (-9)`
- 部分克隆后 `.git/index.lock` 残留
- 缺失对象导致 checkout 不完整

最终采用的恢复路径是：下载当前提交源码包，保留 `.git`，补齐 Git 对象并重建 index。关键操作为：

```bash
tar -tzf /tmp/Visual-Whole-Body-Control-869104c.tar.gz > /tmp/Visual-Whole-Body-Control-869104c.tar.list
git -C Visual-Whole-Body-Control read-tree HEAD
git -C Visual-Whole-Body-Control status --short
git -C Visual-Whole-Body-Control fsck --no-dangling
git -C Visual-Whole-Body-Control ls-files | wc -l
git -C Visual-Whole-Body-Control log -1 --oneline --decorate
```

这些命令的作用：

- `tar -tzf`：验证源码包可完整遍历。
- `git read-tree HEAD`：重建 `.git/index`。
- `git status --short`：确认工作树状态。
- `git fsck --no-dangling`：检查 Git 对象完整性。
- `git ls-files | wc -l`：确认索引中文件数量，结果为 `2043`。
- `git log -1`：确认当前提交，结果为 `869104c (grafted, HEAD -> main, origin/main, origin/HEAD) add ack`。

恢复后的项目源码路径为：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/
```

## 3. 论文和代码理解

早期阅读了 `paper/paper.html`，并按需参考了 `paper/source/` 下的 TeX 源码和 `paper/paper.pdf`。同时阅读了 `vbc/` 与 `Visual-Whole-Body-Control/` 中的 high-level、low-level 代码。

当时形成的核心理解：

- low-level policy 控制 B1+Z1 的移动和末端执行器跟踪。
- high-level state-based teacher 输出末端目标、夹爪和 base command。
- vision student 后续通过 DAgger 从 teacher 学习。
- teacher 训练使用大量并行环境，论文附录中提到 teacher 用 `10240` 并行环境，student 用 `240` 并行环境。

相关分析记录主要在：

```text
/home/hjr/projects/2-Nexus/VBC/new2read.md
/home/hjr/projects/2-Nexus/VBC/agent-log/
```

## 4. 本地低层权重与仿真验证

用户通过 Google Drive 下载了低层控制权重：

```text
/home/hjr/projects/2-Nexus/VBC/model_38000.pt
```

读取与校验命令：

```bash
ls -lh /home/hjr/projects/2-Nexus/VBC/model_38000.pt
file /home/hjr/projects/2-Nexus/VBC/model_38000.pt
sha256sum /home/hjr/projects/2-Nexus/VBC/model_38000.pt
python -c "import torch; x=torch.load('/home/hjr/projects/2-Nexus/VBC/model_38000.pt', map_location='cpu'); print(x.keys(), x.get('iter'))"
```

关键结果：

```text
SHA256: a39f77e5c91298a6e3353664f55caf7afcc2cda4c0fd82ce6659c9fff21a6e23
iter: 38000
checkpoint keys: infos, iter, model_state_dict, optimizer_state_dict
```

为了让项目原加载逻辑找到它，创建了两个链接：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt -> /home/hjr/projects/2-Nexus/VBC/model_38000.pt

/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/data/low_policy/model_38000.pt -> /home/hjr/projects/2-Nexus/VBC/model_38000.pt
```

并将 high-level 配置临时指向：

```yaml
low_policy_path: data/low_policy/model_38000.pt
```

低层回放代表命令：

```bash
python low-level/legged_gym/scripts/play.py \
  --exptid google_drive \
  --task b1z1 \
  --proj_name b1z1-low \
  --checkpoint 38000 \
  --observe_gait_commands \
  --flat_terrain \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --headless
```

低层较长 headless 数值回放结果：

- 运行 `1000` steps
- 进程退出码 `0`
- base 有明显位移，策略不是静止或立即崩溃
- `final_xy_displacement_m=2.45274`
- `approx_xy_path_m=9.32541`

低层视频录制命令：

```bash
python low-level/legged_gym/scripts/play.py \
  --exptid google_drive \
  --task b1z1 \
  --proj_name b1z1-low \
  --checkpoint 38000 \
  --observe_gait_commands \
  --flat_terrain \
  --sim_device cuda:0 \
  --rl_device cuda:0 \
  --headless \
  --record_video
```

低层视频证据：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/videos/google_drive/google_drive-0-38000.mp4
```

视频验证结果：

- 文件约 `654K`
- `250` 帧
- `720x480`
- `25fps`
- 时长约 `10.0s`
- 首尾帧有明显差异，非空白静态视频

## 5. 本机 high-level smoke test 与硬件瓶颈

本机环境修复包括：

- 修复 Isaac Gym 与 NumPy 兼容问题，将 NumPy 调整为 `1.23.5`
- 保证先导入 `isaacgym`，避免 `PyTorch was imported before isaacgym modules`
- 使用项目内 `third_party/isaacgym/python`
- 确认 `skrl`、`rsl_rl`、`legged_gym` 从项目路径导入

high-level teacher 短 rollout 命令：

```bash
python high-level/train_multistate.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps 8 \
  --headless \
  --task B1Z1PickMulti \
  --experiment_dir experiments-smoke \
  --wandb_name smoke-high-level-teacher-rollout8 \
  --debug \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --stop_pick
```

结果：`8/8` rollout 完成，退出码 `0`。

但本机 GTX 1660 SUPER 6GB 在 `--timesteps 24` 后进入 PPO update 时 OOM，因此只能用于调试和小规模 smoke，不能承担完整 teacher 训练。

本地 high-level smoke 产物：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/experiments-smoke/smoke-high-level-teacher-rollout8/
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/experiments-smoke/smoke-high-level-teacher/
```

## 6. 远端训练环境准备

远端目录：

```text
/home/ubuntu/vbc-remote/
```

准备了本地和远端脚本目录：

```text
/home/hjr/projects/2-Nexus/VBC/remote-run/
```

代表性远端准备命令：

```bash
bash remote-run/remote/00_probe_remote.sh
bash remote-run/remote/10_install_env.sh
bash remote-run/remote/20_prepare_project.sh
bash remote-run/remote/30_check_imports.sh
bash remote-run/remote/40_smoke_teacher.sh
bash remote-run/remote/50_start_teacher_tmux.sh
```

这些脚本的作用：

- `00_probe_remote.sh`：确认远端系统、GPU、磁盘等基础状态。
- `10_install_env.sh`：安装或准备 high-level 训练环境。
- `20_prepare_project.sh`：准备项目路径和依赖路径。
- `30_check_imports.sh`：检查 Isaac Gym、PyTorch、skrl、项目模块导入。
- `40_smoke_teacher.sh`：用 `10240` env、`24` timesteps 做 full-scale teacher smoke，确认不 OOM。
- `50_start_teacher_tmux.sh`：在 tmux 后台启动 teacher 长训。

远端 high-level teacher smoke 日志：

```text
/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260530-161843.log
```

smoke 结果：

- `TEACHER_SMOKE_ENVS=10240`
- `TEACHER_SMOKE_TIMESTEPS=24`
- 完成 `24/24`
- 退出码 `0`

## 7. 第一次 teacher 长训：使用 Google Drive `model_38000.pt`

第一次远端 teacher 长训基于用户下载的低层权重 `model_38000.pt`。

核心训练命令形式：

```bash
python train_multistate.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps 60000 \
  --headless \
  --task B1Z1PickMulti \
  --experiment_dir b1-pick-multi-teacher \
  --wandb_name teacher-20260530-162617 \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --stop_pick
```

远端日志：

```text
/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log
```

远端 checkpoint 目录：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/
```

训练完成：

- `60000/60000`
- 训练退出时间：`2026-05-31 12:24:41 CST`
- `agent_60000.pt` 存在
- `best_agent.pt` 存在
- 常规 `agent_*.pt` 共 `120` 个
- checkpoint 目录约 `2.4G`

但评估失败：

- `agent_60000.pt` headless eval：`Total success rate 0.0`
- `agent_58000.pt` headless eval：`Total success rate 0.0`
- `best_agent.pt` 直接 eval 因文件名解析 bug 失败
- 用 `/tmp/best_58000.pt -> best_agent.pt` 绕过后，probe 仍 `new_success=0`

本地已同步的第一次 teacher 权重：

```text
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-20260530-162617/checkpoints/agent_60000.pt
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-20260530-162617/checkpoints/agent_58000.pt
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-20260530-162617/checkpoints/best_agent.pt
```

第一次 teacher 的结论：训练完成但没有学会有效拾取，不适合作为 student 专家。

## 8. 追查 `publiccheckrollrew_42000.pt`

仓库原始 high-level 配置指向：

```text
/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt
```

本地和远端都没有找到该文件。W&B 上能确认 `ericonaldo/b1z1-low` 项目中存在低层 run：

```text
displayName: publiccheckrollrew
run: yf5dxfg1
```

但公开文件和 artifact 中没有 `.pt` checkpoint。直接访问以下路径均返回 `404`：

```text
https://api.wandb.ai/files/ericonaldo/b1z1-low/yf5dxfg1/publiccheckrollrew_42000.pt
https://api.wandb.ai/files/ericonaldo/b1z1-low/yf5dxfg1/model_42000.pt
```

结论：可以确认低层训练 run 公开存在并跑过 `42000/45000`，但无法从当前公开渠道下载作者本地 `publiccheckrollrew_42000.pt`。

## 9. 按作者低层环境重训 low-level

为了接近作者低层策略，远端创建了低层专用环境：

```text
/home/ubuntu/anaconda3/envs/vbc-low-cu113
```

环境特征：

- Python `3.8.18`
- `torch==1.10.0+cu113`
- `torchvision==0.11.1+cu113`
- `torchaudio==0.10.0+cu113`
- `numpy==1.23.5`
- `isaacgym==1.0rc3`
- `rsl-rl==1.0.2`
- `wandb==0.16.2`

低层新增远端脚本：

```text
remote-run/remote/90_install_low_level_env.sh
remote-run/remote/91_smoke_low_level.sh
remote-run/remote/92_start_low_level_tmux.sh
remote-run/remote/93_monitor_low_level.sh
remote-run/remote/94_prepare_teacher_from_low.sh
remote-run/remote/95_wait_low_then_start_teacher.sh
remote-run/remote/96_start_wait_low_then_teacher_tmux.sh
```

低层 smoke：

```bash
LOW_SMOKE_ITERATIONS=2 bash remote-run/remote/91_smoke_low_level.sh
```

结果：两次 PPO iteration 正常完成。

低层长训 run：

```text
publiccheckrollrew-retrain-20260531-2354
```

低层日志：

```text
/home/ubuntu/vbc-remote/remote-logs/publiccheckrollrew-retrain-20260531-2354.low.log
```

低层 run 目录：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/
```

训练中发生过两次外部中断，分别从 `model_23400.pt` 和 `model_34400.pt` 续训。最终完成：

```text
model_42000.pt
model_45000.pt
```

关键低层权重：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_37000.pt
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_42000.pt
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_45000.pt
```

## 10. 第二次 teacher：使用重训低层 `model_42000.pt`

低层 `model_42000.pt` 被准备为 high-level low policy：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_42000.pt
```

对应 teacher run：

```text
teacher-after-publiccheckrollrew-retrain-20260531-2354-42000
```

远端日志：

```text
/home/ubuntu/vbc-remote/remote-logs/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000.teacher.log
```

远端 checkpoint 目录：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/checkpoints/
```

结果：

- 完成 `60000/60000`
- 最终 `agent_60000.pt`
- `best_agent.pt`
- 日志末尾 `Total success rate ≈ 0.00582955`
- probe 中 `new_success=0`

该 run 仍失败。相比第一次训练，成功率略高，但仍不足以作为 teacher。

## 11. 失败诊断：W&B 对比与 `cube_falls` 差异

对比 W&B 官方高层项目 `ericonaldo/b1-pick-multi-teacher` 后发现：

- 成功 run 如 `publiccheckrollrew_37000_2` 的 `Reward / Total reward (mean) ≈ 93.5`
- 多类别成功率约 `0.536-0.881`
- 本地第二次 teacher 的 `Reward / Total reward (mean)` 峰值只有约 `9.513`
- 第二次 teacher 的 termination probe 中 `cube_falls=170/172`

从 W&B 成功 run 保存的 `envs/b1z1_pickmulti.py` 对比出关键差异：

```python
# 官方成功 run 口径
cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))

# 本地/远端失败训练前的严格口径
cube_falls = z_cube < self.table_heights
```

这个差异会把“物体只要低于桌面顶面”就判定掉落，而官方成功 run 允许约 `3.5cm` 容差。考虑到失败 teacher 主要被 `cube_falls` 提前 reset，该差异被判断为高优先级修复点。

修复文件：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py
```

修复后关键代码：

```python
cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))
```

修复后用旧失败 teacher 做 300-step 对照：

- `new_episodes=10`
- `new_success=0`
- 新官方口径 `cube_falls=6`
- 旧严格口径统计 `cube_below_table=163`

这说明 reset 结构确实被修复，但旧策略本身仍不可用。

## 12. 第三次 teacher：官方 `cube_falls` 修复 + `publiccheckrollrew_37000.pt`

W&B 成功 run `publiccheckrollrew_37000_2` 使用 37000 步附近低层 checkpoint，因此本轮将重训低层的 `model_37000.pt` 准备为：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_37000.pt
```

远端配置确认：

```yaml
low_policy_path: data/low_policy/publiccheckrollrew_37000.pt
```

修复后 full-scale smoke：

```text
/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log
```

smoke 条件：

- `10240` env
- `24` timesteps
- 退出码 `0`

第三次 teacher run：

```text
teacher-cubefallfix-low37000-20260604-1530
```

远端日志：

```text
/home/ubuntu/vbc-remote/remote-logs/teacher-cubefallfix-low37000-20260604-1530.teacher.log
```

远端 run 目录：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/
```

训练命令核心：

```bash
CUDA_VISIBLE_DEVICES=1 python train_multistate.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps 60000 \
  --headless \
  --task B1Z1PickMulti \
  --experiment_dir b1-pick-multi-teacher \
  --wandb_name teacher-cubefallfix-low37000-20260604-1530 \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --stop_pick
```

说明：

- `CUDA_VISIBLE_DEVICES=1`：使用远端物理 GPU1。
- `--timesteps 60000`：与 README 和 W&B 成功 run 训练长度对齐。
- `--headless`：远端无 GUI，使用 headless Isaac Gym。
- `--roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick`：与官方 high-level teacher 命令保持一致的核心 flags。
- `VBC_USE_WANDB=0`：实际启动时禁用 W&B，避免远端账号和网络阻塞；这只影响日志上传，不应改变训练逻辑。

训练完成状态：

- `60000/60000`
- `agent_60000.pt` 生成时间：`2026-06-05 11:55:22 CST`
- `best_agent.pt` 生成时间：`2026-06-05 09:36:26 CST`
- `best_agent.pt` 时间戳与 `agent_53000.pt` 对齐
- 远端 GPU 空闲
- 严格日志末尾 `Total success rate ≈ 0.0385234`

## 13. 当前候选 teacher 的效果确认

评估脚本：

```text
/tmp/vbc_termination_probe.py
```

这个脚本会通过 `train_multistate.get_trainer(is_eval=True)` 创建 34 个 eval 环境，记录窗口内新增 episode、新增成功、reset 原因、`lifted_object`、`cube_falls`、`ik_fail` 等指标。

`agent_60000.pt` 300-step probe 命令：

```bash
CUDA_VISIBLE_DEVICES=0 python /tmp/vbc_termination_probe.py \
  --checkpoint /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/agent_60000.pt \
  --steps 300 \
  --experiment-dir /tmp/vbc-termination-probe-cubefallfix-60000 \
  --wandb-name termination-cubefallfix-agent60000-300
```

`agent_60000.pt` 结果：

```text
new_episodes: 40
new_success: 5
window_success_rate: 0.125
lifted_object: 5
lifted_now: 9
cube_falls: 35
cube_below_table: 877
ik_fail: 0
```

`best_agent.pt` 直接评估仍会失败：

```text
ValueError: invalid literal for int() with base 10: 'agent'
```

原因是 `train_multistate.py` 会从 checkpoint 文件名最后一段解析 step 数，`best_agent.pt` 不符合 `agent_<step>.pt` 形式。

解决方法：创建数字后缀 symlink。

远端：

```bash
ln -sfn \
  /home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt \
  /tmp/best_53000.pt
```

使用 `53000` 是因为 `best_agent.pt` 时间戳与 `agent_53000.pt` 对齐。

`best_agent.pt` 300-step probe：

```bash
CUDA_VISIBLE_DEVICES=0 python /tmp/vbc_termination_probe.py \
  --checkpoint /tmp/best_53000.pt \
  --steps 300 \
  --experiment-dir /tmp/vbc-termination-probe-cubefallfix-best \
  --wandb-name termination-cubefallfix-best-300
```

结果：

```text
new_episodes: 12
new_success: 4
window_success_rate: 0.3333333333333333
lifted_object: 4
lifted_now: 5
cube_falls: 8
cube_below_table: 659
ik_fail: 0
```

`best_agent.pt` 1000-step probe：

```bash
CUDA_VISIBLE_DEVICES=0 python /tmp/vbc_termination_probe.py \
  --checkpoint /tmp/best_53000.pt \
  --steps 1000 \
  --experiment-dir /tmp/vbc-termination-probe-cubefallfix-best-1000 \
  --wandb-name termination-cubefallfix-best-1000
```

结果：

```text
new_episodes: 23
new_success: 7
window_success_rate: 0.30434782608695654
lifted_object: 7
lifted_now: 10
cube_falls: 16
cube_below_table: 2202
ik_fail: 0
```

结论：

- `best_agent.pt` 明显优于 `agent_60000.pt`
- `best_agent.pt` 已经出现稳定的真实拾取成功信号
- 当前最强证据是 1000-step probe 的 `7/23` 成功
- 它仍不是官方成功 run 量级，因此建议继续做更长/分类 eval，再决定是否投入 student 长训

## 14. 当前权重与重要产物路径

### 本地候选 teacher 权重

```text
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt
SHA256: adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c
```

文件名兼容 symlink：

```text
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt -> best_agent.pt
```

本地同步配置：

```text
/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/config/b1z1_pickmulti.yaml
```

### 远端第三次 teacher 产物

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/
```

远端 checkpoint：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/agent_60000.pt
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/agent_53000.pt
```

远端训练日志：

```text
/home/ubuntu/vbc-remote/remote-logs/teacher-cubefallfix-low37000-20260604-1530.teacher.log
```

远端 smoke 日志：

```text
/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log
```

### low-level 相关产物

用户下载的初始低层权重：

```text
/home/hjr/projects/2-Nexus/VBC/model_38000.pt
SHA256: a39f77e5c91298a6e3353664f55caf7afcc2cda4c0fd82ce6659c9fff21a6e23
```

低层 headless 视频：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/videos/google_drive/google_drive-0-38000.mp4
```

远端低层重训 run：

```text
/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/
```

远端低层重训关键权重：

```text
model_37000.pt
model_42000.pt
model_45000.pt
```

## 15. 重要问题与注意事项

### 15.1 `best_agent.pt` 文件名解析 bug

`best_agent.pt` 不能直接用于部分 eval/probe 路径，因为脚本会尝试从文件名中解析整数 step：

```python
checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
```

`best_agent.pt` 的最后一段是 `agent`，所以会报：

```text
ValueError: invalid literal for int() with base 10: 'agent'
```

规避方式：

```bash
ln -sfn best_agent.pt best_53000.pt
```

或后续直接修代码，让 `best_agent.pt` 不走整数解析。

### 15.2 当前没有 high-level 可视化视频证据

当前完整视频证据主要是 low-level `model_38000.pt` 的 headless MP4：

```text
/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/videos/google_drive/google_drive-0-38000.mp4
```

high-level teacher 当前主要是 headless 数值 probe 证据，还没有为 `best_agent.pt` 录制可视化 MP4。若下一步需要直观展示，应优先做 high-level teacher 的短视频录制或 viewer 回放。

### 15.3 当前 teacher 不是论文级强专家

当前 `best_agent.pt` 的 1000-step probe 成功率约 `30.4%`，比之前 `0%` 的失败结果大幅改善。但官方成功 run 的多类别成功率约 `0.4-0.9`，因此：

- 可以作为候选 teacher
- 可以考虑 student smoke 或短 DAgger
- 不建议直接投入完整 student 长训并假设最终会成功

## 16. 建议下一步

1. 对 `best_agent.pt` 做更长、更分类的 headless eval，确认哪些物体类别成功、哪些失败。
2. 为 `best_agent.pt` 录制 high-level 可视化视频，补齐直观效果证据。
3. 如果用户接受候选 teacher 的成功率风险，可以先启动小规模 student smoke 或短 DAgger。
4. 如果目标是接近论文/官方 W&B 曲线，应继续排查高层环境版本、seed、低层策略等价性和 reward/curriculum 差异。
