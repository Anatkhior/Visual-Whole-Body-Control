# 项目概览

`Visual-Whole-Body-Control/` 是论文 `Visual Whole-Body Control for Legged Loco-Manipulation` 对应的代码仓库。仓库 README 指向项目主页 `https://wholebody-b1.github.io/` 和 arXiv 论文 `https://arxiv.org/abs/2403.16967`。

顶层结构：

- `high-level/`：视觉运动高层策略、任务环境、DAGGER 训练、网络模块和高层策略运行脚本。
- `low-level/`：通用低层控制器，用于四足机器人和机械臂，任务是跟踪末端执行器目标位姿与机器人速度。
- `third_party/`：项目随仓库包含的第三方依赖源码或安装材料，包括 `isaacgym`、`rsl_rl`、`skrl` 等。
- `teaser.jpg`：README 中使用的项目示意图。

高层策略当前 README 只描述了 picking multiple objects 任务。低层 README 描述通用低层策略训练与回放入口。

本地同步目录：`/home/hjr/projects/2-Nexus/VBC/Visual-Whole-Body-Control/`。
