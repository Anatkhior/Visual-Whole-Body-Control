# 产物索引

## 工作区

- 工作区根目录：`/home/hjr/projects/2-Nexus/VBC/`
- 项目目录：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/`
- 项目日志目录：`/home/hjr/projects/2-Nexus/VBC/agent-log/`
- 远端 SSH 交接信息：`/home/hjr/projects/2-Nexus/VBC/agent-log/SSH.md`
- Teacher 成功候选权重综合实验日志：`/home/hjr/projects/2-Nexus/VBC/teacher_success_log.md`
  - 覆盖范围：项目克隆与恢复、低层权重验证、低层视频证据、远端环境、三轮 teacher 训练、失败诊断、`cube_falls` 修复、当前候选 `best_agent.pt` 与 probe 结果。

## 仓库内续作入口

- 顶层交接说明：`REPRODUCTION_HANDOFF.md`
- 可提交的中文记录：`agent-log/`
- 远端训练脚本：`remote-run/`
- Teacher 阶段汇总：`teacher_success_log.md`
- 私有连接信息模板：`agent-log/SSH.example.md`
- 不提交但后续需要手动准备的内容：低层 `.pt` 权重、远端 `.pt` checkpoint、大 `.log`、视频、`remote-run/local/remote.env`、`agent-log/SSH.md`、`remote-results/`
- 其他机器 `git clone` 后若要同步低层权重，优先在 `remote-run/local/remote.env` 中设置 `LOW_POLICY_LOCAL_PATH`；若要沿当前官方对齐路线，远端准备阶段使用 `LOW_POLICY_TARGET_NAME=publiccheckrollrew_37000.pt`。

## 项目关键文件

- 顶层说明：`Visual-Whole-Body-Control/README.md`
- 高层说明：`Visual-Whole-Body-Control/high-level/README.md`
- 低层说明：`Visual-Whole-Body-Control/low-level/README.md`
- 高层配置：`Visual-Whole-Body-Control/high-level/data/cfg/`
- 高层训练脚本：`Visual-Whole-Body-Control/high-level/train_multistate.py`
- 高层视觉 student 训练脚本：`Visual-Whole-Body-Control/high-level/train_multi_bc_deter.py`
- 低层训练脚本：`Visual-Whole-Body-Control/low-level/legged_gym/scripts/train.py`
- 低层回放脚本：`Visual-Whole-Body-Control/low-level/legged_gym/scripts/play.py`

## 权重与模型

- 用户下载的低层控制权重：`/home/hjr/projects/2-Nexus/VBC/model_38000.pt`
  - 大小：约 `2.0M`
  - SHA256：`a39f77e5c91298a6e3353664f55caf7afcc2cda4c0fd82ce6659c9fff21a6e23`
  - PyTorch checkpoint 键：`infos`、`iter`、`model_state_dict`、`optimizer_state_dict`
  - `iter=38000`
- 低层回放链接：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/b1z1-low/google_drive/model_38000.pt -> /home/hjr/projects/2-Nexus/VBC/model_38000.pt`
- 高层低层策略链接：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/data/low_policy/model_38000.pt -> /home/hjr/projects/2-Nexus/VBC/model_38000.pt`
- 已修改的高层配置：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
  - `low_policy_path: "data/low_policy/model_38000.pt"`

## 视频证据

- 低层 headless 回放视频：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/low-level/logs/videos/google_drive/google_drive-0-38000.mp4`
  - 生成时间：2026-05-26 11:04 左右。
  - 文件大小：`669454` bytes，约 `654K`。
  - 解码信息：`250` 帧，`720x480`，`25fps`，duration `10.0` 秒，codec `h264`，pix_fmt `yuv420p`。
  - 内容验证：首帧标准差 `44.608`，抽样帧平均差异最高 `15.305`，首尾帧平均差异 `14.505`，首尾变化像素比例 `0.17102`。

## 高层 smoke test 产物

- 高层标准入口 `24` timestep 诊断产物目录：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/experiments-smoke/smoke-high-level-teacher/`
  - `b1z1_pickmulti.yaml`：`6378` bytes。
  - `events.out.tfevents.1779766535.hjr-MS-7D45.2845005.0`：`88` bytes。
  - 该运行在 PPO update 阶段 CUDA OOM，保留为失败诊断证据。
- 高层标准入口 `8` timestep 短 rollout 产物目录：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/experiments-smoke/smoke-high-level-teacher-rollout8/`
  - `b1z1_pickmulti.yaml`：`6378` bytes。
  - `events.out.tfevents.1779767039.hjr-MS-7D45.2852727.0`：`88` bytes。
  - 该运行完成 `8/8` timestep，退出码 `0`。

## 远端训练启动包

- 远端启动包根目录：`/home/hjr/projects/2-Nexus/VBC/remote-run/`
- 使用说明：`remote-run/README.md`
- 本地脚本：
  - `remote-run/local/remote.env.example`
  - `remote-run/local/generate_ssh_key.sh`
  - `remote-run/local/sync_to_remote.sh`
  - `remote-run/local/fetch_results.sh`
- 远端脚本：
  - `remote-run/remote/00_probe_remote.sh`
  - `remote-run/remote/10_install_env.sh`
  - `remote-run/remote/20_prepare_project.sh`
  - `remote-run/remote/30_check_imports.sh`
  - `remote-run/remote/40_smoke_teacher.sh`
  - `remote-run/remote/50_start_teacher_tmux.sh`
  - `remote-run/remote/60_monitor_teacher.sh`
  - `remote-run/remote/70_start_student_tmux.sh`
  - `remote-run/remote/80_collect_results.sh`
  - `remote-run/remote/common.sh`

## 远端运行产物

- 远端工作目录：`/home/ubuntu/vbc-remote/`
- 远端 smoke 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260530-161843.log`
- 远端 teacher 长训日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-20260530-162617.teacher.log`
- 远端 teacher tmux 会话：`vbc_teacher_g0`
- 远端同步后的项目根目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/`
- 远端 teacher checkpoint 目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/`
- 远端最终 teacher checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/agent_60000.pt`
- 远端 best teacher checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/checkpoints/best_agent.pt`
- 远端 TensorBoard event：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-20260530-162617/events.out.tfevents.1780129902.3090ubuntu20-04.411124.0`
- 远端临时 teacher 探针脚本：`/tmp/vbc_teacher_probe.py`
- 远端临时 TensorBoard 解析脚本：`/tmp/vbc_parse_tb.py`
- 远端临时 best symlink：`/tmp/best_58000.pt -> .../checkpoints/best_agent.pt`

## 本地同步的远端产物

- 本地 teacher 结果目录：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-20260530-162617/`
- 本地 teacher checkpoints：
  - `remote-results/teacher-20260530-162617/checkpoints/agent_60000.pt`
  - `remote-results/teacher-20260530-162617/checkpoints/agent_58000.pt`
  - `remote-results/teacher-20260530-162617/checkpoints/best_agent.pt`
- 本地/远端一致的 teacher checkpoint SHA256：
  - `best_agent.pt`：`d140cca286eb4107348aed2b4596ee1858a88bfb7bc389ced208a1de27af8299`
  - `agent_58000.pt`：`9367999f2566fa128418b60a737c9d764b37f4363a97623d7533f6f01abb3f42`
  - `agent_60000.pt`：`1ffdbdd1bbeeb3b1e72e447964943c8e86e77cf90a1e19da9d6944025d66e4e4`
- 本地 teacher eval 日志：
  - `remote-results/teacher-20260530-162617/logs/teacher-eval-agent60000-20260531-131714.log`
  - `remote-results/teacher-20260530-162617/logs/teacher-eval-agent58000-20260531-132711.log`
  - `remote-results/teacher-20260530-162617/logs/teacher-eval-best-agent-20260531-132628.log`
- 部分训练日志：
  - `remote-results/teacher-20260530-162617/logs/teacher-20260530-162617.teacher.log.partial`
  - 该文件是中断的部分同步，不是完整日志。

## 临时恢复文件

- 源码包：`/tmp/Visual-Whole-Body-Control-869104c.tar.gz`
- 源码包列表：`/tmp/Visual-Whole-Body-Control-869104c.tar.list`
- 陈旧锁文件备份：`/tmp/Visual-Whole-Body-Control.index.lock.stale`

## 外部低层 checkpoint 线索

- README 公开低层权重链接：`https://drive.google.com/file/d/1KIfKu77QkrwbK-YllSWclqb6vJknGgjv/view?usp=sharing`
- 本地当前可用低层权重：`/home/hjr/projects/2-Nexus/VBC/model_38000.pt`
- 高层当前低层权重链接：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/data/low_policy/model_38000.pt`
- 原始高层配置期望路径：`/data/mhliu/visual_wholebody/high-level/data/low_policy/publiccheckrollrew_42000.pt`
- W&B 公开低层项目：`https://wandb.ai/ericonaldo/b1z1-low`
- W&B `publiccheckrollrew` run：`https://wandb.ai/ericonaldo/b1z1-low/runs/yf5dxfg1`
- W&B 公开文件 API：
  - `https://api.wandb.ai/files/ericonaldo/b1z1-low/yf5dxfg1/wandb-summary.json`
  - `https://api.wandb.ai/files/ericonaldo/b1z1-low/yf5dxfg1/requirements.txt`
  - `https://api.wandb.ai/files/ericonaldo/b1z1-low/yf5dxfg1/output.log`
- `/tmp/vbc_publiccheckrollrew_output.log` 是本轮为了检索 W&B `output.log` 下载的临时分析文件，不作为长期项目产物依赖。

## 低层重训产物

- 本地新增远端脚本：
  - `remote-run/remote/90_install_low_level_env.sh`
  - `remote-run/remote/91_smoke_low_level.sh`
  - `remote-run/remote/92_start_low_level_tmux.sh`
  - `remote-run/remote/93_monitor_low_level.sh`
  - `remote-run/remote/94_prepare_teacher_from_low.sh`
  - `remote-run/remote/95_wait_low_then_start_teacher.sh`
  - `remote-run/remote/96_start_wait_low_then_teacher_tmux.sh`
- 远端低层环境：`/home/ubuntu/anaconda3/envs/vbc-low-cu113`
- 远端低层长训 run：`publiccheckrollrew-retrain-20260531-2354`
- 远端低层 tmux：`vbc_low_g0`
- 远端低层 run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354`
- 远端低层 log：`/home/ubuntu/vbc-remote/remote-logs/publiccheckrollrew-retrain-20260531-2354.low.log`
- 远端已生成初始 checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_0.pt`
- 远端低层关键 checkpoint：
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_42000.pt`
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_45000.pt`
- 高层已准备的低层策略：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_42000.pt`
- 远端 watcher tmux：`vbc_wait_low_teacher`
- 远端 watcher log：`/home/ubuntu/vbc-remote/remote-logs/wait-publiccheckrollrew-retrain-20260531-2354-then-teacher.log`
- watcher 目标低层 checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/low-level/logs/b1z1-low/publiccheckrollrew-retrain-20260531-2354/model_42000.pt`
- watcher 将启动的 teacher run：`teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`

## 新低层权重对应的 teacher 重训产物

- 远端 teacher tmux：`vbc_teacher_g1`
- 远端 teacher run：`teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`
- 远端 teacher log：`/home/ubuntu/vbc-remote/remote-logs/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000.teacher.log`
- 远端 teacher run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000`
- 远端 teacher checkpoint 目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/checkpoints`
- 远端最终 teacher checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/checkpoints/agent_60000.pt`
- 远端 best teacher checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-after-publiccheckrollrew-retrain-20260531-2354-42000/checkpoints/best_agent.pt`

## 官方 `cube_falls` 修复后 teacher 重训产物

- 本地修复文件：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - 关键判定：`cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))`
- 远端修复文件：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
- 远端当前高层配置：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
  - `low_policy_path: data/low_policy/publiccheckrollrew_37000.pt`
- 远端当前低层策略：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_37000.pt`
- 修复后 teacher smoke 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-cubefallfix-low37000-20260604-152424.log`
- 修复后 teacher tmux：`vbc_teacher_g1`
- 修复后 teacher run：`teacher-cubefallfix-low37000-20260604-1530`
- 修复后 teacher log：`/home/ubuntu/vbc-remote/remote-logs/teacher-cubefallfix-low37000-20260604-1530.teacher.log`
- 修复后 teacher run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530`
- 修复后 teacher checkpoint 目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints`
- 修复后 teacher 首个 checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/agent_500.pt`
  - 大小：`20622586` bytes
  - 生成时间：`2026-06-04 15:35:41 CST`
- 当前 student 使用的候选 teacher：
  - 远端最佳权重：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`
  - 本地同步权重：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`
  - SHA256：`adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c`
  - 注意：远端 `agent_53000.pt` 与 `best_agent.pt` 哈希不同，不能作为等价替代。
- 修复后 teacher 最终 checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/agent_60000.pt`
  - 生成时间：`2026-06-05 11:55:22 CST`
- 修复后 teacher 候选 best：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`
  - 生成时间：`2026-06-05 09:36:26 CST`
  - 时间戳与 `agent_53000.pt` 对齐。
- 本地同步的候选 best：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_agent.pt`
  - SHA256：`adb524f339ee24857d9e7d95424276b89c6f6f3b679c8e0526983125cfbd976c`
- 本地 best 文件名兼容 symlink：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt -> best_agent.pt`
- 本地同步的 run 配置：`/home/hjr/projects/2-Nexus/VBC/remote-results/teacher-cubefallfix-low37000-20260604-1530/config/b1z1_pickmulti.yaml`
- 临时 termination probe：`/tmp/vbc_termination_probe.py`

## 当前 student 训练产物

- 本地修复文件：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/train_multi_bc_deter.py`
  - 关键改动：student 创建环境时设置 `cfg["enableCameraSensors"] = True`，并为 headless camera sensors 提供非负 `graphics_device_id`。
- 本地修复脚本：`/home/hjr/projects/2-Nexus/VBC/remote-run/remote/70_start_student_tmux.sh`
  - 关键改动：支持 `STUDENT_DISPLAY`，当前远端 student 必须使用 `STUDENT_DISPLAY=:0`；2026-06-06 20:56 CST 起还支持记录 `STUDENT_CHECKPOINT` 和 Python 退出码。
- 本地远端通用脚本：`/home/hjr/projects/2-Nexus/VBC/remote-run/remote/common.sh`
  - 关键改动：2026-06-06 20:56 CST 起支持 `STUDENT_CHECKPOINT`，用于给 student 训练入口追加显式 `--checkpoint`。
- 远端 teacher symlink：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt -> best_agent.pt`
- student 失败 smoke 日志：
  - `/home/ubuntu/vbc-remote/remote-logs/student-smoke-best53000-20260606-1029.log`
  - `/home/ubuntu/vbc-remote/remote-logs/student-smoke-best53000-camerafix-20260606-1049.log`
  - `/home/ubuntu/vbc-remote/remote-logs/student-smoke-debug-gpu0-camerafix-20260606-1120.log`
- student 成功 smoke 日志：
  - `/home/ubuntu/vbc-remote/remote-logs/student-smoke-debug-display0-gpu0-20260606-1135.log`
  - `/home/ubuntu/vbc-remote/remote-logs/student-smoke-display0-gpu0-24step-20260606-1142.log`
- 当前 student 长训：
  - tmux：`vbc_student_g0`
  - run：`student-best53000-display0-20260606-1145`
  - log：`/home/ubuntu/vbc-remote/remote-logs/student-best53000-display0-20260606-1145.student.log`
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145`
  - teacher checkpoint path：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-cubefallfix-low37000-20260604-1530/checkpoints/best_53000.pt`
  - 2026-06-06 18:50 CST 旧训练段停在：`10159/60000`，无正常退出标记
  - 2026-06-06 21:00 CST 已从 `agent_10000.pt` 续训，续训段进度条显示 `279/50000`，约等效 `10279/60000`
  - 2026-06-07 13:57 CST 续训完成：续训段 `50000/50000`，等效总进度 `60000/60000`，Python exit code `0`
  - 当前 checkpoint 目录：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145/checkpoints`
  - 已生成：`agent_1000.pt` 至 `agent_60000.pt` 的常规 checkpoint，以及 `best_agent.pt`
  - 最终 student checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145/checkpoints/agent_60000.pt`
  - 当前 best student checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145/checkpoints/best_agent.pt`
  - 最终 checkpoint 单文件大小：`101222307` bytes，约 `96.5 MiB`
  - 远端 `agent_60000.pt` SHA256：`fd7f1e0cf0406ed41b02b3e52737a9bc340f5a786d0329b21cd5d0167ac346c7`
  - 远端 `best_agent.pt` SHA256：`053e21f8a94fa6acdac1b30854350ff71a3a0bc63cf6783bb2674e6e19e2d69e`
  - 远端 eval symlink：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-stu/student-best53000-display0-20260606-1145/checkpoints/best_50000.pt -> best_agent.pt`
- 本地已同步 student 归档目录：`/home/hjr/projects/2-Nexus/VBC/remote-results/student-best53000-display0-20260606-1145/`
  - 配置：`config/b1z1_pickmulti.yaml`
  - 训练日志：`logs/student-best53000-display0-20260606-1145.student.log`
  - `agent_60000.pt` headless eval 日志：`logs/student-eval-agent60000-headless420s-20260607-1421.log`
  - `best_agent.pt` headless eval 日志：`logs/student-eval-best50000-headless420s-20260607-1502.log`
  - 视频录制失败日志：`logs/student-eval-agent60000-record150-20260607-1402.log`
  - 本地已同步日志 SHA256 与远端一致：
    - `student-best53000-display0-20260606-1145.student.log`：`7cc443f035bde10f40c51d20e62004f4dfe5e0158ea8c9685ec68f4264c5545a`
    - `student-eval-agent60000-headless420s-20260607-1421.log`：`79fdf174d0aa3e94ea3ae662a79a992c0b4dfb83d357e7d44de1a6c8b3a583ab`
    - `student-eval-best50000-headless420s-20260607-1502.log`：`5154f45555e44a218a0009e210ac74dd6ca68a04bef59970114f5c2bfa6ed1f0`
    - `student-eval-agent60000-record150-20260607-1402.log`：`407637171f95d9e7d0aacfd7a4b0e0a8fae04cb234575af40a0608629a2e9dde`
  - 注意：本地 `checkpoints/` 目前没有完整 student checkpoint；慢速传输留下的不完整文件已删除。

## 官方对齐后 teacher 重训产物

- 本地对齐文件：
  - `/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - `/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
  - `/home/hjr/projects/2-Nexus/VBC/remote-run/remote/50_start_teacher_tmux.sh`
- W&B 成功 run 保存文件参考：
  - `/tmp/wandb_97yfb8x1_b1z1_pickmulti.py`
  - `/tmp/wandb_97yfb8x1_b1z1_pickmulti.yaml`
  - `/tmp/wandb_97yfb8x1_train_multistate.py`
- 远端对齐文件：
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/envs/b1z1_pickmulti.py`
  - `/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/cfg/b1z1_pickmulti.yaml`
  - `/home/ubuntu/vbc-remote/remote-run/remote/50_start_teacher_tmux.sh`
- 远端低层策略：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/data/low_policy/publiccheckrollrew_37000.pt`
  - 大小：约 `2.0M`
  - SHA256：`b61e2167ea3c370668d17bbd35b641703313a0f6bc57e1dc5117c5073af69fd6`
- 对齐后 teacher smoke 日志：`/home/ubuntu/vbc-remote/remote-logs/teacher-smoke-10240env-24step-20260607-211143.log`
- 对齐后 teacher 长训：
  - tmux：`vbc_teacher_g1`
  - run：`teacher-officialalign-low37000-20260607-2116`
  - log：`/home/ubuntu/vbc-remote/remote-logs/teacher-officialalign-low37000-20260607-2116.teacher.log`
  - run dir：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-officialalign-low37000-20260607-2116`
  - 首个 checkpoint：`/home/ubuntu/vbc-remote/Visual-Whole-Body-Control/high-level/b1-pick-multi-teacher/teacher-officialalign-low37000-20260607-2116/checkpoints/agent_500.pt`
  - 2026-06-07 21:34:59 CST 状态：进度约 `515/60000`，训练仍在运行。
