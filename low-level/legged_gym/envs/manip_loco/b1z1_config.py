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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np


# 这个文件只定义配置，不直接启动仿真、不直接训练。
# task_registry.register("b1z1", ManipLoco, B1Z1RoughCfg(), B1Z1RoughCfgPPO(), "b1z1")
# 会把下面两个配置类交给训练入口使用：
# B1Z1RoughCfg 负责环境、机器人、观测、奖励、地形、随机化；
# B1Z1RoughCfgPPO 负责策略网络、PPO 算法和训练运行器流程。
class B1Z1RoughCfg(LeggedRobotCfg):
    # 机械臂末端执行器目标配置。
    # 这里的 ee 是末端执行器的缩写，也就是机械臂末端。
    # ManipLoco 会周期性采样一个末端目标，让机械臂去跟踪它。
    class goal_ee:
        # 末端目标命令维度。这里是 3 维球坐标目标。
        num_commands = 3
        # 每次末端目标从起点插值到终点所需时间范围，单位是秒。
        traj_time = [1, 3]
        # 到达目标后保持一段时间，再重新采样目标，单位是秒。
        hold_time = [0.5, 2]
        # 末端目标轨迹碰撞检查的上边界，目标轨迹落入该盒子范围时会被判为不可用。
        collision_upper_limits = [0.1, 0.2, -0.05]
        # 末端目标轨迹碰撞检查的下边界，和上边界一起定义一个需要避开的空间盒子。
        collision_lower_limits = [-0.8, -0.2, -0.7]
        # 末端目标不能低于这个 z 高度，否则认为目标在地下，不可用。
        underground_limit = -0.7
        # 在起点到终点之间采样多少个中间点做碰撞和地下检查。
        num_collision_check_samples = 10
        # 目标表示方式。sphere 表示用球坐标采样末端位置。
        command_mode = 'sphere'
        # 根据球坐标的俯仰角推导默认末端姿态时使用的补偿项。
        # 在 manip_loco.py 里会用于 default_pitch = -pos_p + arm_induced_pitch。
        arm_induced_pitch = 0.38

        # 球坐标中心点相对机器人和地形的位置。
        # 末端目标不是绕世界原点采样，而是绕机器人基座附近的一个中心采样。
        class sphere_center:
            # 中心点相对机器人基座朝前偏移 0.3 米。
            x_offset = 0.3
            # 中心点相对机器人基座左右方向不偏移。
            y_offset = 0
            # 中心点高度相对地形固定抬高 0.7 米。
            z_invariant_offset = 0.7

        # 末端目标的采样范围。
        # pos_l、pos_p、pos_y 可以理解为球坐标里的距离、俯仰角、偏航角。
        class ranges:
            # 回合初始化时，末端目标插值的起始球坐标。
            init_pos_start = [0.5, np.pi / 8, 0]
            # 回合初始化时，末端目标插值的结束球坐标。
            init_pos_end = [0.7, 0, 0]
            # 末端目标距离中心点的距离范围。
            pos_l = [0.4, 0.95]
            # 末端目标俯仰角范围。
            pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            # 末端目标偏航角范围。
            pos_y = [-1.2, 1.2]

            # 末端姿态相对默认姿态的横滚角扰动范围。
            delta_orn_r = [-0.5, 0.5]
            # 末端姿态相对默认姿态的俯仰角扰动范围。
            delta_orn_p = [-0.5, 0.5]
            # 末端姿态相对默认姿态的偏航角扰动范围。
            delta_orn_y = [-0.5, 0.5]
            # 末端跟踪奖励的目标阈值或参考值，当前文件里保留该字段。
            final_tracking_ee_reward = 0.55

        # 球坐标位置误差的缩放系数。这里三个维度都不额外缩放。
        sphere_error_scale = [1, 1, 1]
        # 姿态误差缩放系数。这里三个姿态维度也不额外缩放。
        orn_error_scale = [1, 1, 1]

    # 观测噪声配置。
    # 打开后，会在观测上叠加随机噪声，用来提高策略鲁棒性。
    class noise:
        # 当前关闭观测噪声。
        add_noise = False
        # 全局噪声强度，会整体缩放下面各类噪声。
        noise_level = 1.0

        # 不同观测量对应的噪声尺度。
        class noise_scales:
            # 关节位置噪声尺度。
            dof_pos = 0.01
            # 关节速度噪声尺度。
            dof_vel = 1.5
            # 机器人基座线速度噪声尺度。
            lin_vel = 0.1
            # 机器人基座角速度噪声尺度。
            ang_vel = 0.2
            # 重力方向观测噪声尺度。
            gravity = 0.05
            # 高度测量点噪声尺度。
            height_measurements = 0.1

    # 底盘运动命令配置。
    # 低层策略要同时跟踪机器人基座速度命令和机械臂末端目标。
    class commands:
        # 是否使用课程学习逐步改变命令难度。
        curriculum = True
        # 命令维度。这里通常对应 x 方向速度、y 方向速度或占位、偏航角速度。
        num_commands = 3
        # 每隔多少秒重新采样一次底盘速度命令。
        resampling_time = 3.

        # x 方向线速度命令范围或权重的调度参数。
        lin_vel_x_schedule = [0, 0.5]
        # 偏航角速度命令范围或权重的调度参数。
        ang_vel_yaw_schedule = [0, 1]
        # 偏航角速度跟踪奖励的调度参数。
        tracking_ang_vel_yaw_schedule = [0, 1]

        # 判断偏航命令是否足够大、是否属于行走状态时使用的阈值。
        ang_vel_yaw_clip = 0.5
        # 判断 x 方向速度命令是否足够大、是否属于行走状态时使用的阈值。
        lin_vel_x_clip = 0.2

        # 速度命令采样范围。
        class ranges:
            # x 方向线速度范围，单位 m/s。
            lin_vel_x = [-0.8, 0.8]
            # 偏航角速度范围，单位 rad/s。
            ang_vel_yaw = [-1.0, 1.0]

    # 观测和动作的归一化、裁剪配置。
    class normalization:
        # 不同观测项进入策略前的缩放系数。
        class obs_scales:
            # 机器人基座线速度缩放。
            lin_vel = 1.0
            # 机器人基座角速度缩放。
            ang_vel = 1.0
            # 关节位置缩放。
            dof_pos = 1.0
            # 关节速度通常数值较大，所以这里缩小到 0.05。
            dof_vel = 0.05
            # 地形高度测量缩放。
            height_measurements = 5.0

        # 观测最终会被裁剪到 [-100, 100]，防止异常值直接进入网络。
        clip_observations = 100.
        # 动作也会被裁剪到 [-100, 100]；实际有效幅度还会被 action_scale 缩放。
        clip_actions = 100.

    # 环境总体尺寸和观测动作维度配置。
    class env:
        # 并行仿真环境数量。PPO 每轮会同时从这些环境收集数据。
        num_envs = 6144
        # 策略输出动作维度：12 个腿部关节 + 6 个机械臂关节。
        # 注意：ManipLoco.step() 里会把 actions[:, 12:] 置零，机械臂主要通过末端目标和 IK 控制。
        num_actions = 12 + 6
        # 力矩相关维度，同样覆盖 12 个腿部关节和 6 个机械臂关节。
        num_torques = 12 + 6
        # 动作延迟步数；-1 表示不使用动作延迟。
        action_delay = 3
        # 夹爪关节数量，观测里会从部分关节向量中排除夹爪。
        num_gripper_joints = 1
        # 当前时刻本体感知观测维度：
        # 2 维身体姿态 + 3 维基座角速度 + 18 维关节位置 + 18 维关节速度
        # + 12 维上一时刻腿部动作 + 4 维足端接触 + 3 维基座命令
        # + 3 维末端局部笛卡尔目标 + 3 维末端姿态占位。
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
        # 特权信息维度：5 个质量参数 + 1 个摩擦系数 + 12 个腿部电机强度偏差。
        num_priv = 5 + 1 + 12
        # 历史观测长度。策略除了当前观测，还会看到过去 10 帧本体感知。
        history_len = 10
        # 策略输入总维度：当前本体感知 + 历史本体感知 + 特权信息。
        num_observations = num_proprio * (history_len + 1) + num_priv
        # 如果不为 None，step() 会额外返回评价器使用的 privileged_obs_buf；
        # 当前为 None，表示策略网络和评价网络都走当前 obs_buf 体系。
        num_privileged_obs = None
        # 是否把超时信息发送给算法，用于区分超时结束和失败终止。
        send_timeouts = True
        # 单个回合的最长时间，单位秒。
        episode_length_s = 10
        # 是否重排关节顺序，使观测和动作顺序符合代码约定。
        reorder_dofs = True
        # 是否使用键盘遥操作命令；teleop.py 会覆盖这个值。
        teleop_mode = False
        # 是否录制视频。
        record_video = False
        # 是否进入待机模式；待机时命令会被清零。
        stand_by = False
        # 是否把步态相位相关命令加入观测。打开后 num_proprio 会在 ManipLoco.__init__ 中额外增加 5。
        observe_gait_commands = False
        # 步态或周期相关控制频率参数。
        frequencies = 2

    # 初始状态配置，继承基础腿式机器人配置后覆盖 B1+Z1 的具体默认姿态。
    class init_state(LeggedRobotCfg.init_state):
        # 机器人根节点初始位置，单位米。
        pos = [0.0, 0.0, 0.5]
        # 动作为 0 时的默认关节角，单位 rad。
        # 策略动作会在这个默认角基础上叠加 action_scale * action。
        default_joint_angles = {
            # 左前腿。
            'FL_hip_joint': 0.2,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,

            # 左后腿。
            'RL_hip_joint': 0.2,
            'RL_thigh_joint': 0.8,
            'RL_calf_joint': -1.5,

            # 右前腿。
            'FR_hip_joint': -0.2,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,

            # 右后腿。
            'RR_hip_joint': -0.2,
            'RR_thigh_joint': 0.8,
            'RR_calf_joint': -1.5,

            # Unitree Z1 机械臂默认姿态。
            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -0.63,
            'z1_wrist_angle': -0.84,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': 1.57,
            'z1_jointGripper': -0.785,
        }
        # 重置时随机偏航角的范围。
        rand_yaw_range = np.pi / 2
        # 重置时环境原点或初始位置扰动范围。
        origin_perturb_range = 0.5
        # 重置时初始速度扰动范围。
        init_vel_perturb_range = 0.1

    # PD 控制和动作缩放配置。
    class control:
        # PD 刚度 Kp。joint 对腿部关节，z1 对机械臂关节。
        stiffness = {'joint': 80, 'z1': 5}
        # PD 阻尼 Kd。joint 对腿部关节，z1 对机械臂关节。
        damping = {'joint': 2.0, 'z1': 0.5}

        # 是否让策略自适应调整机械臂控制增益。
        adaptive_arm_gains = False
        # 动作缩放：目标关节角 = 默认关节角 + action_scale * action。
        # 前 12 个数对应四条腿，每条腿 3 个关节；后 6 个数对应机械臂。
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        # 控制降采样。策略每输出一次动作，底层仿真会执行 decimation 个小步。
        decimation = 4
        # 是否启用机械臂力矩监督损失。
        torque_supervision = False

    # 机器人资产配置，告诉 Isaac Gym 加载哪个 URDF，以及哪些连杆用于脚、夹爪和接触判断。
    class asset(LeggedRobotCfg.asset):
        # B1+Z1 机器人 URDF 路径。
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/b1z1.urdf'
        # 脚部连杆名称中包含的关键词。
        foot_name = "foot"
        # 夹爪末端连杆名称。
        gripper_name = "ee_gripper_link"
        # 这些部位发生接触会被奖励函数惩罚。
        penalize_contacts_on = ["thigh", "trunk", "calf"]
        # 这些部位发生接触会直接终止回合；当前为空，表示不因指定连杆接触而终止。
        terminate_after_contacts_on = []
        # 自碰撞开关，0 表示启用自碰撞，1 表示禁用自碰撞。
        self_collisions = 0
        # 是否翻转可视化附件。
        flip_visual_attachments = False
        # 是否折叠 URDF 中的固定关节。
        collapse_fixed_joints = True
        # 是否固定基座连杆。False 表示机器人基座可以自由运动。
        fix_base_link = False

    # 小方块对象配置。这个低层文件保留了方块配置，主要用于带物体的扩展场景。
    class box:
        # 方块边长。
        box_size = 0.1
        # 是否随机化方块的质量。
        randomize_base_mass = True
        # 方块附加质量范围。
        added_mass_range = [-0.001, 0.050]
        # 方块相对环境原点的 x 位置。
        box_env_origins_x = 0
        # 方块相对环境原点的 y 位置采样范围。
        box_env_origins_y_range = [0.1, 0.3]
        # 方块中心 z 位置。
        box_env_origins_z = box_size / 2 + 0.16

    # 机械臂控制相关配置。
    class arm:
        # 初始末端目标在机器人基座坐标系下的位置。
        init_target_ee_base = [0.2, 0.0, 0.2]
        # 抓取时末端和目标之间的偏移。
        grasp_offset = 0.08
        # 操作空间控制的刚度，前三个是位置，后三个是姿态。
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        # 操作空间控制的阻尼，按临界阻尼形式从 kp 推出。
        osc_kd = 2 * (osc_kp ** 0.5)

    # 域随机化配置。
    # 训练时随机化物理参数，让策略不要只适应一个固定仿真世界。
    class domain_rand:
        # 是否把随机化后的特权参数放进观测。
        observe_priv = True
        # 是否随机化地面摩擦。
        randomize_friction = True
        # 地面摩擦系数采样范围。
        friction_range = [0.3, 3.0]
        # 是否随机化机器人基座质量。
        randomize_base_mass = True
        # 机器人基座附加质量范围，单位 kg。
        added_mass_range = [0., 15.]
        # 是否随机化机器人基座质心位置。
        randomize_base_com = True
        # 机器人基座质心 x 方向偏移范围。
        added_com_range_x = [-0.15, 0.15]
        # 机器人基座质心 y 方向偏移范围。
        added_com_range_y = [-0.15, 0.15]
        # 机器人基座质心 z 方向偏移范围。
        added_com_range_z = [-0.15, 0.15]
        # 是否随机化电机强度。
        randomize_motor = True
        # 腿部电机强度乘法系数范围。
        leg_motor_strength_range = [0.7, 1.3]
        # 机械臂电机强度乘法系数范围。
        arm_motor_strength_range = [0.7, 1.3]
        # 是否随机化夹爪质量。
        randomize_gripper_mass = True
        # 夹爪附加质量范围。
        gripper_added_mass_range = [0, 0.1]
        # 是否定期给机器人施加外部推动。
        push_robots = True
        # 每隔多少秒推动一次机器人。
        push_interval_s = 8
        # 推动时 xy 平面的最大速度扰动。
        max_push_vel_xy = 0.5

    # 奖励配置。
    # 这里不直接写奖励函数，只声明使用哪个奖励容器，以及每个奖励项的权重。
    class rewards:
        # 奖励函数容器名称，对应 envs/rewards/maniploco_rewards.py。
        reward_container_name = "maniploco_rewards"

        # 通用奖励参数。
        # False 表示总奖励可以为负；True 会把负总奖励裁剪为 0。
        only_positive_rewards = False
        # 机器人基座速度跟踪奖励使用的误差尺度，常见形式是 exp(-error^2 / sigma)。
        tracking_sigma = 0.2
        # 末端执行器跟踪奖励使用的误差尺度。
        tracking_ee_sigma = 1
        # 关节位置软限制比例，超过 URDF 限制的这个比例后开始惩罚。
        soft_dof_pos_limit = 1.
        # 关节速度软限制比例。
        soft_dof_vel_limit = 1.
        # 力矩软限制比例。
        soft_torque_limit = 0.4
        # 机器人基座目标高度。
        base_height_target = 0.55
        # 接触力超过这个值会受到惩罚。
        max_contact_force = 40.

        # 步态控制奖励参数。
        # 足端速度相关步态奖励的误差尺度。
        gait_vel_sigma = 0.5
        # 足端接触力相关步态奖励的误差尺度。
        gait_force_sigma = 0.5
        # 步态概率或相位塑形的集中程度参数。
        kappa_gait_probs = 0.07
        # 足端摆动高度目标。
        feet_height_target = 0.3

        # feet_air_time 是否统计全部脚。False 时部分实现可能只统计前脚。
        feet_aritime_allfeet = False
        # feet_height 是否统计全部脚。
        feet_height_allfeet = False

        # 奖励权重说明：
        # 权重为 0 的项仍可能被计算和记录，只是对总奖励没贡献；
        # 权重为 None 的项通常表示不计算、不记录。
        # 注意：Python 类体里同名变量后写的值会覆盖前写的值。
        class scales:
            # 步态控制奖励。
            # 只有 observe_gait_commands 为 True 时，这两个接触塑形奖励才有实际意义。
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0
            # 鼓励脚离地时间合适，帮助形成步态。
            feet_air_time = 2.0
            # 鼓励摆动脚达到一定高度。
            feet_height = 1.0

            # 底盘速度跟踪奖励。
            # 鼓励 x 方向线速度接近命令。
            tracking_lin_vel_max = 2.0
            # L1 形式的 x 速度跟踪项，当前权重为 0。
            tracking_lin_vel_x_l1 = 0.
            # 指数形式的 x 速度跟踪项，当前权重为 0。
            tracking_lin_vel_x_exp = 0
            # 鼓励偏航角速度跟踪命令。
            tracking_ang_vel = 0.5

            # 下面几项是早期或重复保留的权重；后面同名项会覆盖前面的同名赋值。
            delta_torques = -1.0e-7 / 4.0
            work = 0
            energy_square = 0.0
            # 直接惩罚力矩大小。
            torques = -2.5e-5
            # 站立时鼓励不要乱动。
            stand_still = 1.0
            # 行走时鼓励合适的关节行为。
            walking_dof = 1.5
            # 默认关节位置惩罚项，当前权重为 0。
            dof_default_pos = 0.0
            # 关节误差惩罚项，当前权重为 0。
            dof_error = 0.0
            # 存活奖励。
            alive = 1.0
            # 惩罚机器人基座 z 方向速度，减少上下弹跳。
            lin_vel_z = -1.5
            # 惩罚横滚姿态过大。
            roll = -2

            # 通用惩罚项。
            # 惩罚 x/y 平面外的角速度。
            ang_vel_xy = -0.2
            # 惩罚关节加速度，减少高频抖动。
            dof_acc = -7.5e-7
            # 惩罚不希望发生的碰撞。
            collision = -10.
            # 惩罚动作变化过快。
            action_rate = -0.015
            # 惩罚关节位置接近或超过限制。
            dof_pos_limits = -10.0
            # 覆盖前面的 delta_torques，最终生效的是这个值。
            delta_torques = -1.0e-7
            # 惩罚髋关节偏离期望范围。
            hip_pos = -0.3
            # 覆盖前面的 work，最终生效的是这个值。
            work = -0.003
            # 惩罚足端 jerk，减少脚部运动突变。
            feet_jerk = -0.0002
            # 惩罚脚拖地。
            feet_drag = -0.08
            # 惩罚过大的足端接触力。
            feet_contact_forces = -0.001
            # 姿态惩罚项，当前总项权重为 0。
            orientation = 0.0
            # 行走状态姿态惩罚项，当前权重为 0。
            orientation_walking = 0.0
            # 站立状态姿态惩罚项，当前权重为 0。
            orientation_standing = 0.0
            # 惩罚机器人基座高度偏离目标。
            base_height = -5.0
            # 行走时力矩惩罚项，当前权重为 0。
            torques_walking = 0.0
            # 站立时力矩惩罚项，当前权重为 0。
            torques_standing = 0.0
            # 覆盖前面的 energy_square，当前权重仍为 0。
            energy_square = 0.0
            # 行走状态能耗平方惩罚，当前权重为 0。
            energy_square_walking = 0.0
            # 站立状态能耗平方惩罚，当前权重为 0。
            energy_square_standing = 0.0
            # 行走状态机器人基座高度惩罚，当前权重为 0。
            base_height_walking = 0.0
            # 站立状态机器人基座高度惩罚，当前权重为 0。
            base_height_standing = 0.0
            # 惩罚 y 方向线速度，当前权重为 0。
            penalty_lin_vel_y = 0.

        # 机械臂相关奖励权重。
        class arm_scales:
            # 机械臂失败终止奖励，None 表示不启用。
            arm_termination = None
            # 球坐标末端跟踪，当前权重为 0。
            tracking_ee_sphere = 0.
            # 世界坐标末端跟踪，当前主要启用这一项。
            tracking_ee_world = 0.8
            # 行走状态下的球坐标末端跟踪，当前权重为 0。
            tracking_ee_sphere_walking = 0.0
            # 站立状态下的球坐标末端跟踪，当前权重为 0。
            tracking_ee_sphere_standing = 0.0
            # 笛卡尔坐标末端跟踪，None 表示不启用。
            tracking_ee_cart = None
            # 机械臂姿态奖励，None 表示不启用。
            arm_orientation = None
            # 机械臂能耗绝对值惩罚，None 表示不启用。
            arm_energy_abs_sum = None
            # 末端姿态跟踪，当前权重为 0。
            tracking_ee_orn = 0.
            # 只看横滚角和偏航角或部分姿态的末端跟踪项，None 表示不启用。
            tracking_ee_orn_ry = None

    # Isaac Gym 查看器默认相机位置。
    class viewer:
        # 查看器相机所在位置，单位米。
        pos = [-20, 0, 20]
        # 查看器相机看向的位置，单位米。
        lookat = [0, 0, -2]

    # 回合终止阈值配置。
    class termination:
        # 横滚角绝对值超过该阈值时终止。
        r_threshold = 0.8
        # 俯仰角绝对值超过该阈值时终止。
        p_threshold = 0.8
        # 机器人基座高度低于该阈值时终止。
        z_threshold = 0.1

    # 地形配置。
    class terrain:
        # 地形网格类型。trimesh 表示用三角网格地形。
        mesh_type = 'trimesh'
        # 高度场转三角网格的方法，fast 是更快的选项名。
        hf2mesh_method = "fast"
        # fast 方法允许的最大近似误差。
        max_error = 0.1
        # 地形水平分辨率，越小越精细但计算越慢。
        horizontal_scale = 0.05
        # 地形垂直分辨率。
        vertical_scale = 0.005
        # 地形边界尺寸，单位米。
        border_size = 25
        # 粗糙地形高度范围。
        height = [0.00, 0.1]
        # 间隙地形的间隙大小范围。
        gap_size = [0.02, 0.1]
        # 踏脚石地形的石块间距范围。
        stepping_stone_distance = [0.02, 0.08]
        # 降采样后的高度场尺度。
        downsampled_scale = 0.075
        # 是否启用地形课程学习。
        curriculum = False

        # 是否生成全竖直地形结构。
        all_vertical = False
        # 是否去掉平地。True 表示倾向于不使用完全平坦地形。
        no_flat = True

        # 地形静摩擦系数。
        static_friction = 1.0
        # 地形动摩擦系数。
        dynamic_friction = 1.0
        # 地形碰撞弹性恢复系数。
        restitution = 0.

        # 是否测量机器人周围的地形高度。
        measure_heights = True
        # x 方向高度采样点，覆盖机器人前后约 1.6 米范围，不含中心线说明来自原注释。
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # y 方向高度采样点。
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        # 是否只选择一种指定地形。
        selected = False
        # selected=True 时传给指定地形生成器的参数。
        terrain_kwargs = None
        # 地形课程学习的初始难度等级。
        max_init_terrain_level = 5
        # 单块地形长度。
        terrain_length = 8.
        # 单块地形宽度。
        terrain_width = 8.
        # 地形行数，通常表示不同难度等级。
        num_rows = 10
        # 地形列数，通常表示不同地形类型。
        num_cols = 20

        # 各类地形的采样权重。当前只有粗糙平地对应权重为 1.0。
        terrain_dict = {"smooth slope": 0.,
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0.,
                        "rough stairs down": 0.,
                        "discrete": 0.,
                        "stepping stones": 0.,
                        "gaps": 0.,
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        # 将 terrain_dict 的权重转成列表，供地形生成逻辑使用。
        terrain_proportions = list(terrain_dict.values())
        # 只对 trimesh 有意义。超过该坡度阈值的斜面会被修正成竖直面；None 表示不启用。
        slope_treshold = None
        # 是否把地形原点 z 坐标强制设为 0。
        origin_zero_z = False


# PPO 训练配置。
# 这个类不会创建环境，它会被 task_registry.make_alg_runner() 读取，
# 再传给 rsl_rl 的 OnPolicyRunner、ActorCritic 和 PPO。
class B1Z1RoughCfgPPO(LeggedRobotCfgPPO):
    # 随机种子。task_registry.get_cfgs() 会把这个 seed 同步给 env_cfg.seed。
    seed = 1
    # 训练运行器类名。当前使用常规 OnPolicyRunner。
    runner_class_name = 'OnPolicyRunner'

    # 策略网络配置。
    class policy:
        # 续训时是否沿用上一个检查点的动作标准差。
        continue_from_last_std = True
        # 初始动作分布标准差。前 12 个对应腿，后 6 个对应机械臂。
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        # 策略网络主干隐藏层维度。
        actor_hidden_dims = [128]
        # 价值网络主干隐藏层维度。
        critic_hidden_dims = [128]
        # 激活函数，可选 elu、relu、selu、crelu、lrelu、tanh、sigmoid。
        activation = 'elu'
        # 策略网络输出是否再经过 tanh。False 表示不在这里强行压到 [-1, 1]。
        output_tanh = False

        # 腿部动作输出头的隐藏层维度。
        leg_control_head_hidden_dims = [128, 128]
        # 机械臂动作输出头的隐藏层维度。
        arm_control_head_hidden_dims = [128, 128]

        # 特权信息编码器的隐藏层和输出维度配置。
        priv_encoder_dims = [64, 20]

        # 腿部动作维度。
        num_leg_actions = 12
        # 机械臂动作维度。
        num_arm_actions = 6

        # 是否启用机械臂自适应增益，跟环境控制配置保持一致。
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains
        # 自适应增益输出的缩放系数。
        adaptive_arm_gains_scale = 10.0

    # PPO 算法配置。
    class algorithm:
        # 价值损失权重。
        value_loss_coef = 1.0
        # 是否使用裁剪后的价值损失。
        use_clipped_value_loss = True
        # PPO 裁剪参数，限制新旧策略概率比的变化幅度。
        clip_param = 0.2
        # 熵奖励权重。0 表示不额外鼓励探索熵。
        entropy_coef = 0.0
        # 每次采样后，对同一批数据训练多少轮。
        num_learning_epochs = 5
        # 每轮训练切成多少个小批次。
        # 小批次大小约等于 num_envs * num_steps_per_env / num_mini_batches。
        num_mini_batches = 4
        # 学习率。
        learning_rate = 2e-4
        # 学习率调度方式。fixed 是固定学习率的选项名。
        schedule = 'fixed'
        # 折扣因子 gamma。
        gamma = 0.99
        # GAE 的 lambda 参数，用于优势函数估计。
        lam = 0.95
        # 目标 KL 散度。None 表示不使用自适应 KL 控制。
        desired_kl = None
        # 梯度裁剪上限。
        max_grad_norm = 1.
        # 策略动作分布标准差的下限，防止探索噪声塌缩到过小。
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        # 价值混合调度参数，PPO 中用于控制某些混合项随训练迭代变化。
        mixing_schedule = [1.0, 0, 3000]
        # 是否启用力矩监督，和 control.torque_supervision 保持一致。
        torque_supervision = B1Z1RoughCfg.control.torque_supervision
        # 力矩监督权重调度参数。
        torque_supervision_schedule = [0.0, 1000, 1000]
        # 是否启用机械臂自适应增益，和 control.adaptive_arm_gains 保持一致。
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains
        # 历史编码器和特权隐变量对齐的更新频率。
        dagger_update_freq = 20
        # 特权正则项系数调度。
        # 在 PPO.update() 中会从 0 逐步增加到 0.1，起始迭代约为 3000，过渡长度约为 7000。
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000]

    # 训练运行器配置，控制一次训练跑多久、每轮收集多少数据、如何保存模型。
    class runner:
        # 策略网络类名，对应 rsl_rl.modules.ActorCritic。
        policy_class_name = 'ActorCritic'
        # 算法类名，对应 rsl_rl.algorithms.PPO。
        algorithm_class_name = 'PPO'
        # 每个并行环境每轮采样多少步。
        num_steps_per_env = 24
        # 最大策略更新次数。
        max_iterations = 45000
        # 每隔多少次迭代检查并保存一次检查点。
        save_interval = 200
        # 实验名，会影响日志目录组织。
        experiment_name = 'b1z1_v2'
        # 运行名称，空字符串表示由外部参数或默认规则决定。
        run_name = ''
        # 是否从已有检查点继续训练。
        resume = False
        # 续训时加载哪个运行记录，-1 表示最后一个运行记录。
        load_run = -1
        # 续训时加载哪个检查点，-1 表示最后保存的模型。
        checkpoint = -1
        # 直接指定续训路径时使用；通常由 load_run 和 checkpoint 推导。
        resume_path = None
