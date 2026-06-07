# 项目记录入口

本目录记录当前工作区中 `Visual-Whole-Body-Control/` 的同步、环境、验证和后续维护信息。所有后续记录以中文为主；文件名、命令、变量、路径、论文名、仓库名等原本为英文的内容保持原样。

## 阅读顺序

1. 先读 `01-current-state.md`，确认当前状态和下一步。
2. 需要理解项目来源与结构时读 `10-project-overview.md`。
3. 需要复现实验环境或运行命令时读 `20-runtime-and-integration.md`、`50-implementation-or-training.md`。
4. 需要检查同步证据、失败路径或完整命令时读 `raw-by-topic/` 中对应主题。

## 路由

| 需求 | 文件 |
| --- | --- |
| 当前状态、阻塞、下一步 | `01-current-state.md` |
| 项目背景、目录结构 | `10-project-overview.md` |
| 克隆、下载、环境安装 | `20-runtime-and-integration.md` |
| 数据、资产、模型权重 | `40-datasets-and-artifacts.md` |
| 训练和运行入口 | `50-implementation-or-training.md` |
| 验证命令与结果 | `60-evaluation-and-validation.md` |
| 失败尝试、诊断、不要重复的路径 | `70-diagnostics-and-dead-ends.md` |
| 重要路径索引 | `80-artifacts-index.md` |
| 证据来源和日志覆盖范围 | `90-source-map.md` |

## 维护规则

- 每次实质性工作结束后更新 `01-current-state.md`。
- 只把稳定结论放入摘要文件；长命令、长输出、失败细节写入 `raw-by-topic/*.raw.md`。
- 记录中的不确定性必须明确标注，不把推测写成事实。
- 不删除原始仓库内容；如需改动项目代码，先记录目标和验证方式。
