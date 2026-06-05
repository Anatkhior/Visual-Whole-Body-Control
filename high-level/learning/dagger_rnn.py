from typing import Any, Dict, Optional, Tuple, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

# [start-config-dict-torch]
# RNN 版 DAgger 的默认配置。这个文件和 dagger.py 的核心目标一致：
# 用 student 观测重新前向 student policy，并让 student 动作通过 MSE 拟合 teacher_actions。
# 不同点是这里还会存储/采样 RNN hidden state，让 GRU student 能按时间序列训练。
DAGGER_DEFAULT_CONFIG = {
    "rollouts": 16,                 # 每收集多少个环境 step 后更新一次 student；训练入口会覆盖为 24。
    "learning_epochs": 8,           # 每次更新时，把 memory 中的数据重复训练多少轮。
    "mini_batches": 2,              # 每轮训练把 memory 切成多少个 mini-batch。

    "discount_factor": 0.99,        # 兼容 skrl/PPO 的折扣因子字段；当前 DAgger MSE 更新不计算 return。
    "lambda": 0.95,                 # 兼容 skrl/PPO 的 GAE lambda 字段；当前 DAgger MSE 更新不计算 advantage。

    "learning_rate": 1e-3,                  # Adam 优化器学习率；训练入口会覆盖为 5e-5。
    "learning_rate_scheduler": None,        # 可选学习率调度器；当前训练入口将其设为 None。
    "learning_rate_scheduler_kwargs": {},   # 学习率调度器参数；只有 learning_rate_scheduler 非 None 时才会使用。

    "state_preprocessor": None,             # 可选观测预处理器；当前 RNN DAgger 更新中未实际启用。
    "state_preprocessor_kwargs": {},        # 观测预处理器参数。
    "value_preprocessor": None,             # 兼容 skrl/PPO 的 value 预处理器字段；当前没有 value 网络。
    "value_preprocessor_kwargs": {},        # value 预处理器参数。

    "random_timesteps": 0,          # 前多少步随机动作；训练入口设为 0，表示 student 从一开始就用网络输出动作。
    "learning_starts": 0,           # 训练从第几步后开始；训练入口设为 0，表示 memory 满一个 rollouts 后即可更新。

    "grad_norm_clip": 0.5,              # 梯度范数裁剪阈值，用于避免一次更新过大。
    "ratio_clip": 0.2,                  # 兼容 skrl/PPO 的 ratio clip 字段；当前 DAgger loss 不使用。
    "value_clip": 0.2,                  # 兼容 skrl/PPO 的 value clip 字段；当前 DAgger loss 不使用。
    "clip_predicted_values": False,     # 兼容 skrl/PPO 的 value 裁剪开关；当前 DAgger loss 不使用。

    "entropy_loss_scale": 0.0,      # 熵正则权重；默认 0，通常只训练 MSE imitation loss。
    "value_loss_scale": 1.0,        # 兼容 skrl/PPO 的 value loss 权重；当前没有 value loss。

    "kl_threshold": 0,              # 兼容旧 PPO/IL 实验的 KL 早停阈值；当前 KL 计算代码被注释。

    "rewards_shaper": None,         # 兼容 skrl 的 reward shaping 字段；当前 DAgger loss 不直接使用 reward。
    "time_limit_bootstrap": False,  # 兼容 skrl/PPO 的 timeout bootstrap 字段；当前没有 value bootstrap。

    "experiment": {
        "directory": "",            # 实验日志和 checkpoint 的父目录。
        "experiment_name": "",      # 实验名。
        "write_interval": 250,      # TensorBoard/W&B 写日志间隔。

        "checkpoint_interval": 1000,        # 保存 checkpoint 的步数间隔。
        "store_separately": False,          # 是否把 checkpoint 分开保存。

        "wandb": False,             # 是否使用 Weights & Biases 记录训练。
        "wandb_kwargs": {}          # wandb 初始化参数。
    }
}
# [end-config-dict-torch]


class DAgger_RNN(Agent):
    # 默认的 student 训练类。和 DAgger 类相比，这里额外处理 GRU/RNN 的 hidden state：
    # rollout 时把 hidden state 随着环境步推进；训练时从 memory 里按 sequence_length 采样连续片段。
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 state_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        
        """
        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        
        _cfg = copy.deepcopy(DAGGER_DEFAULT_CONFIG)  # 复制默认配置，避免修改全局默认字典。
        _cfg.update(cfg if cfg is not None else {})  # 用训练入口传入的 cfg 覆盖默认值。
        super().__init__(models=models, memory=memory, observation_space=observation_space,
                         action_space=action_space, device=device, cfg=_cfg)  # 初始化 skrl.Agent 基类。
        
        self.state_space = state_space  # student_obs 的空间；wrapper 用 states["states"] 存 student 观测。
        
        # models
        self.policy = self.models.get("policy", None)  # 需要训练的 GRU student policy。
        
        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy  # 告诉 skrl 保存 checkpoint 时包含 student policy 参数。
        
        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]  # 每次更新重复训练 memory 数据的轮数。
        self._mini_batches = self.cfg["mini_batches"]  # 每轮更新的数据切分数量。
        self._rollouts = self.cfg["rollouts"]  # 每隔多少个交互 step 触发一次 _update。
        self._rollout = 0  # 已累计的交互 step 计数器。

        self._grad_norm_clip = self.cfg["grad_norm_clip"]  # 反向传播后的梯度裁剪阈值。
        self._ratio_clip = self.cfg["ratio_clip"]  # 兼容字段；当前 DAgger loss 不使用。
        self._value_clip = self.cfg["value_clip"]  # 兼容字段；当前没有 value loss。
        self._clip_predicted_values = self.cfg["clip_predicted_values"]  # 兼容字段；当前没有 value prediction。

        self._value_loss_scale = self.cfg["value_loss_scale"]  # 兼容字段；当前没有 value loss。
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]  # 熵正则权重；默认 0。

        self._kl_threshold = self.cfg["kl_threshold"]  # KL 早停阈值；当前 KL 相关代码未启用。

        self._learning_rate = self.cfg["learning_rate"]  # Adam 学习率。
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]  # 可选学习率调度器。

        self._state_preprocessor = self.cfg["state_preprocessor"]  # 可选观测预处理器；当前更新中未实际启用。

        self._discount_factor = self.cfg["discount_factor"]  # 兼容字段；当前 DAgger loss 不使用 return。
        self._lambda = self.cfg["lambda"]  # 兼容字段；当前 DAgger loss 不使用 advantage。

        self._random_timesteps = self.cfg["random_timesteps"]  # 随机动作阶段长度；入口通常设为 0。
        self._learning_starts = self.cfg["learning_starts"]  # 多少 step 后开始更新；入口通常设为 0。

        self._rewards_shaper = self.cfg["rewards_shaper"]  # 兼容字段；当前 DAgger loss 不直接使用 reward。
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]  # 兼容字段；当前没有 value bootstrap。
        
        self._fixed_base = self.cfg["fixed_base"]  # 固定底盘训练时，loss 忽略最后 2 个底盘动作维度。
        self._reach_only = self.cfg["reach_only"]  # 只训练 reach 相关动作时，loss 采用特定动作维度组合。
        
        if self.policy is not None:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)  # 只优化 student policy 参数。
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])  # 如果配置了调度器，就绑定 optimizer。
            self.checkpoint_modules["optimizer"] = self.optimizer  # checkpoint 中保存 optimizer 状态，方便断点续训。
        
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])  # 构造观测预处理器。
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor  # checkpoint 中保存预处理器状态。
        else:
            self._state_preprocessor = self._empty_preprocessor  # 没有预处理器时使用 skrl 的空预处理函数。
            
    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)  # 初始化 skrl.Agent 的日志、checkpoint、memory 等内部结构。
        self.set_mode("eval")  # 采样动作时默认 eval；真正更新时 post_interaction 会临时切到 train。
        
        if self.memory is not None:
            # self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="student_obs", size=self.state_space, dtype=torch.float32)  # student 输入；DAgger 更新时会喂给 GRU student。
            self.memory.create_tensor(name="teacher_obs", size=self.observation_space, dtype=torch.float32)  # teacher 输入；主要为了完整记录 transition。
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)  # student 当时输出的动作；可用于调试或替代 loss。
            self.memory.create_tensor(name="teacher_actions", size=self.action_space, dtype=torch.float32)  # teacher 动作标签；MSE 的监督目标。
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)  # 环境奖励；当前 DAgger loss 不直接使用。
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)  # episode 结束标记；训练序列中遇到 done 时要重置 hidden state。
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)  # 策略 log_prob；当前 deterministic policy 时可能为 0。
            
            # tensors sampled during training
            self._tensor_names = ["student_obs", "teacher_obs", "actions", "teacher_actions", "rewards", "terminated", "log_prob"]  # _update 会按这个顺序从 memory 采样。
        
        # RNN specifications
        self._rnn = False # flag to indicate whether RNN is available；没有 policy specification 时保持 False。
        self._rnn_tensors_names = [] # used for sampling during training；保存 memory 中 hidden state tensor 的名字。
        self._rnn_final_states = {"policy": []}  # 最近一次 policy.act 输出的 hidden state。
        self._rnn_initial_states = {"policy": []}  # 下一次 policy.act 要输入的 hidden state。
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)  # 训练时每条序列的长度，来自 student policy。
        
        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True  # 只要 specification 里声明了 RNN hidden size，就启用 RNN 逻辑。
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True)  # 保存第 i 层 RNN hidden state。
                self._rnn_tensors_names.append(f"rnn_policy_{i}")  # 训练采样时根据这个名字取 hidden state。
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))  # 每个并行环境初始 hidden state 为 0。
            
        # create temporary variables needed for storage and computation
        self._current_log_prob = None  # 临时保存最近一次 act 的 log_prob，record_transition 时写入 memory。
        self._current_next_states = None  # 临时保存 next_states；当前文件中主要用于兼容 skrl agent 结构。
        
    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}  # RNN policy 需要把上一时刻 hidden state 一起传入。
        
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": states, **rnn}, role="policy")  # 可选随机探索；本项目配置通常不会进入。
        
        # sample stochastic actions
        actions, log_prob, outputs = self.policy.act({"states": states, **rnn}, role="policy")  # 用 student policy 根据 student_obs 和 hidden state 输出动作。
        if log_prob is None:
            log_prob = 0  # DeterministicMixin 可能不返回 log_prob；用 0 占位，便于 memory 字段形状统一。
        self._current_log_prob = log_prob  # 记录给 record_transition 使用。
        
        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])  # 取出本次前向传播后的 hidden state。
            self._rnn_initial_states["policy"] = self._rnn_final_states["policy"]  # 下一步 rollout 继续使用这个 hidden state。
        
        return actions, log_prob, outputs  # Trainer 会取返回元组的第 0 项作为 student 动作。
    
    def record_transition(self, student_obs: torch.Tensor, teacher_obs, actions: torch.Tensor, teacher_actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: Any, timestep: int, timesteps: int) -> None:
        super().record_transition(states=teacher_obs, actions=actions, rewards=rewards, next_states=next_states,
                                  terminated=terminated, truncated=truncated, infos=infos, timestep=timestep,
                                  timesteps=timesteps)  # 调用 skrl 基类记录通用 transition 信息和日志。
        
        if self.memory is not None:
            self._current_next_states = next_states  # 保存最近的 next_states；保留给 skrl agent 兼容逻辑。
            
            # package RNN states
            rnn_states = {}  # 准备把 hidden state 作为额外字段写入 memory。
            if self._rnn:
                rnn_states.update({f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_final_states["policy"])})  # skrl memory 需要 batch/env 维在前，所以转置 hidden state。
            
            self.memory.add_samples(student_obs=student_obs, teacher_obs=teacher_obs, actions=actions, teacher_actions=teacher_actions, rewards=rewards,
                                    terminated=terminated, log_prob=self._current_log_prob, **rnn_states)  # 把 student_obs、teacher_actions 和 hidden state 聚合到 DAgger 数据集。
        # update RNN states
        if self._rnn:
            # reset states if the episodes have ended
            finished_episodes = terminated.nonzero(as_tuple=False)  # 找出哪些并行环境的 episode 已结束。
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0  # episode 结束的环境必须把 hidden state 清零，避免新 episode 继承旧记忆。
            
            self._rnn_initial_states = self._rnn_final_states  # 更新下一步 rollout 要用的 hidden state。
            
    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1  # 每完成一次环境交互就累加；达到 _rollouts 间隔时开始一次监督学习更新。
        
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")  # 切到训练模式，GRU 会按训练序列处理输入。
            self._update(timestep, timesteps)  # 用 memory 中聚合的数据训练 student 拟合 teacher。
            self.set_mode("eval")  # 更新后切回 eval，继续按单步 rollout 方式采样动作。
        
        # wirte tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)  # 交给 skrl 处理日志写入和 checkpoint 保存。
        
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        
        rnn_policy = {}  # 非 RNN 时保持空字典；RNN 时会放入采样到的 hidden state 和 terminated 标记。
        
            
        cumulative_dagger_loss = 0  # 累计本次更新内所有 mini-batch 的 imitation loss。
        cumulative_entropy_loss = 0  # 累计可选 entropy loss；默认权重为 0。
        
        # learning epochs
        for epoch in range(self._learning_epochs):  # 多次遍历同一批聚合数据，提高样本利用率。
            # compute returns and advantages
            sampled_batches = self.memory.sample_all(names=self._tensor_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)  # 按序列长度采样 transition 数据。
            
            if self._rnn:
                sampled_rnn_batches = self.memory.sample_all(names=self._rnn_tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)  # 同步采样对应的 hidden state 序列。
        
            # kl_divergences = []
            
            # mini-batches loop
            for i, (sampled_student_obs, sampled_teacher_obs, sampled_actions, sampled_teacher_actions, sampled_rewards, sampled_terminated, sampled_log_prob) in enumerate(sampled_batches):
                
                if self._rnn:
                    rnn_policy = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "policy" in n], "terminated": sampled_terminated}  # 还原 GRU 需要的 hidden state 维度，并把 done 标记交给 policy.compute。

                # _, next_log_prob, _ = self.policy.act({"states": sampled_student_obs, "taken_actions": sampled_actions, **rnn_policy}, role="policy")
                
                # # compute approximate KL divergence
                # with torch.no_grad():
                #     ratio = next_log_prob - sampled_log_prob
                #     kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                #     kl_divergences.append(kl_divergence)
                    
                # early stopping with KL divergence
                # if self._kl_threshold and kl_divergence > self._kl_threshold:
                #     break
                
                 # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()  # 可选熵正则；鼓励动作分布更分散。
                else:
                    entropy_loss = 0  # 默认没有熵正则，只有 DAgger MSE。
                
                # compute policy loss
                student_actions, _, _ = self.policy.act({"states": sampled_student_obs, **rnn_policy}, role="policy")  # 用采样到的 student_obs 序列重新前向计算 student 当前动作。
                # student_actions_gripper = student_actions[:, 6]
                # sampled_teacher_actions_gripper = sampled_teacher_actions[:, 6]
                # gripper_loss = F.binary_cross_entropy(student_actions_gripper, sampled_teacher_actions_gripper)
                # student_actions_other = torch.cat([student_actions[:, :6], student_actions[:, 7:]], dim=1)
                # sampled_teacher_actions_other = torch.cat([sampled_teacher_actions[:, :6], sampled_teacher_actions[:, 7:]], dim=1)
                # dagger_loss = F.mse_loss(student_actions_other, sampled_teacher_actions_other) + 0.01 * gripper_loss
                if self._fixed_base:
                    dagger_loss = F.mse_loss(student_actions[:, :-2], sampled_teacher_actions[:, :-2]) # TODO: only for fixed base robot；固定底盘时忽略最后 2 个底盘动作维度。
                elif self._reach_only:
                    dagger_loss = F.mse_loss(student_actions[:, :-3], sampled_teacher_actions[:, :-3]) + \
                        F.mse_loss(student_actions[:, -2:], sampled_teacher_actions[:, -2:])  # reach_only 时跳过倒数第 3 维，但保留末尾 2 维一起监督。
                else:
                    dagger_loss = F.mse_loss(student_actions, sampled_teacher_actions)  # 标准 DAgger loss：student 动作和 teacher 动作逐维均方误差。
                
                # optimization step
                self.optimizer.zero_grad()  # 清空上一轮 mini-batch 留下的梯度。
                (dagger_loss + entropy_loss).backward()  # 反向传播；核心梯度来自 teacher_actions 的 MSE 监督。
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)  # 裁剪梯度，防止单次更新过大。
                self.optimizer.step()  # Adam 根据梯度更新 student policy 参数。
                
                # update cumulative losses
                cumulative_dagger_loss += dagger_loss.item()  # item() 把单元素 tensor 转成 Python 数字，方便日志统计。
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()  # 熵正则启用时记录它的累计值。
            
            # # update learning rate
            # if self._learning_rate_scheduler:
            #     if isinstance(self.scheduler, KLAdaptiveLR):
            #         self.scheduler.step(torch.tensor(kl_divergences).mean())
            #     else:
            #         self.scheduler.step()
        
        # record data
        self.track_data("Loss / DAgger loss", cumulative_dagger_loss / (self._learning_epochs * self._mini_batches))  # 记录本次更新的平均 DAgger MSE。
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))  # 记录平均熵正则项。
        
        if self._current_log_prob != 0:
            self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())  # 随机策略才有意义；确定性策略通常不会记录。
        
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])  # 记录当前学习率。
