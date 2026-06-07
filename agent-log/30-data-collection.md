# 数据采集

当前没有执行新的数据采集、仿真采样或训练日志拉取。

仓库自带的资源主要位于：

- `high-level/data/asset/`：高层任务资产，包括机器人、对象模型、纹理、点云或特征文件等。
- `high-level/data/cfg/`：高层任务配置，例如 `b1z1_pickmulti.yaml`、`b1z1_float.yaml`。
- `low-level/resources/`：低层控制相关机器人资源。

后续如生成新的采样数据、示教数据或仿真结果，应在本文件记录数据来源、命令、时间、输出路径和质量检查方式；长输出写入 `raw-by-topic/30-data-collection.raw.md`。
