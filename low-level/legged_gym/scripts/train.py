# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# NumPy 是原始训练脚本栈保留下来的依赖；这个文件本身没有直接调用它。
# 这里保留 import，是为了不改变已有训练入口的依赖形态。
import numpy as np
# os 用来在训练开始前创建本次实验的日志目录。
import os
# datetime 来自上游脚本模板；当前这个文件没有直接使用它。
from datetime import datetime
# Isaac Gym 通常需要先于许多仿真相关模块导入，避免底层绑定初始化顺序出问题。
import isaacgym

# 项目路径常量：
# LEGGED_GYM_ROOT_DIR 指向 low-level；
# LEGGED_GYM_ENVS_DIR 指向 low-level/legged_gym/envs。
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
# 导入 envs 的关键作用不是直接使用某个名字，而是触发 envs/__init__.py。
# 这个 __init__.py 会把 "b1z1" 这样的任务名注册到 task_registry。
# 如果没有这一步，--task b1z1 就无法映射到 ManipLoco、B1Z1RoughCfg 和 B1Z1RoughCfgPPO。
from legged_gym.envs import *
# get_args 负责解析命令行参数；task_registry 负责创建环境和训练器。
from legged_gym.utils import get_args, task_registry
# torch 是下游 runner 和模型代码会用到的核心库；当前 train() 函数本身没有直接调用它。
import torch
# wandb 用来记录训练指标，并把关键源码文件随实验一起存档。
import wandb

def train(args):
    # 拼出本次实验专属的日志路径，例如：
    # low-level/logs/b1z1-low/<实验ID>。
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        # 创建目录，后续 PPO runner 会把 checkpoint 和日志写到这里。
        os.makedirs(log_pth)
    except:
        # 如果目录已经存在，就继续执行。
        # 注意：这种写法也会吞掉其他 mkdir 错误，是原脚本保留下来的行为。
        pass

    # debug 模式会缩小并行仿真规模，并关闭 wandb 在线记录。
    # 这样可以用更少环境快速检查训练流程是否能跑通。
    if args.debug:
        mode = "disabled"
        # 调试时的可视化网格布局：6 行、2 列。
        args.rows = 6
        args.cols = 2
        # 正式低层训练通常使用几千个并行环境；debug 只用 128 个。
        args.num_envs = 128
    else:
        # 正常训练时，指标会在线同步到 wandb。
        mode = "online"

    # 启动一条 wandb 运行记录。项目名和运行名来自命令行参数，例如：
    # --proj_name b1z1-low 和 --exptid <运行名称>。
    wandb.init(project=args.proj_name, name=args.exptid, mode=mode, dir=LEGGED_GYM_ENVS_DIR +"/logs")

    # 把本次训练使用的关键配置和环境实现保存到 wandb，方便之后追溯 checkpoint 来源。
    # 初学者拿到某个模型权重时，应该优先对照这两个文件。
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/b1z1_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/manip_loco/manip_loco.py", policy="now")

    # 根据 args.task 创建向量化仿真环境。
    # 对 --task b1z1 来说，envs/__init__.py 会把它映射到：
    # ManipLoco + B1Z1RoughCfg。
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # 创建训练 runner。
    # 对 --task b1z1 来说，这里会读取 B1Z1RoughCfgPPO，
    # 用它配置 PPO 算法、ActorCritic 网络、checkpoint 保存方式和最大迭代次数。
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args)

    # 正式开始 PPO 训练。
    # max_iterations 来自 train_cfg.runner；
    # init_at_random_ep_len=True 可以避免所有环境在第 0 次迭代时完全同步 reset。
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # 解析命令行参数，例如 --task、--proj_name、--exptid、
    # --sim_device、--rl_device、--debug、--observe_gait_commands。
    args = get_args()
    # 把解析后的参数交给上面的 train() 函数，进入完整训练流程。
    train(args)
