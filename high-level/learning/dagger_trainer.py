from typing import List, Optional, Union

import copy
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.trainers.torch import Trainer
from skrl.envs.wrappers.torch import Wrapper

DAGGER_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # 总训练步数；DAgger 会在这些步内不断采样、聚合数据、更新 student。
    "headless": False,              # 是否关闭渲染；训练逻辑不依赖画面，只影响是否显示仿真窗口。
    "disable_progressbar": False,   # 是否关闭 tqdm 进度条；None 时由 skrl/终端环境判断。
    "close_environment_at_exit": True,   # 程序正常退出时是否关闭仿真环境。
    "teacher_pretrain": False,      # 是否启用 teacher warm-up；本项目实际用 pretrain_timesteps 控制 warm-up 长度。
}

class DAggerTrainer(Trainer):
    # 这个 Trainer 只负责 DAgger 的“在线采样流程”：
    # 1. 同一时刻分别让 teacher 和 student 根据各自观测输出动作；
    # 2. warm-up 阶段用 teacher 动作推进环境，之后用 student 动作推进环境；
    # 3. 无论是谁推进环境，都把 teacher 动作保存成监督标签，供 DAgger agent 用 MSE 训练 student。
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 teacher_agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        
        _cfg = copy.deepcopy(DAGGER_TRAINER_DEFAULT_CONFIG)  # 复制默认配置，避免修改全局默认字典。
        _cfg.update(cfg if cfg is not None else {})  # 用调用方传入的 cfg 覆盖默认配置。
        agents_scope = agents_scope if agents_scope is not None else []  # skrl 的多 agent 范围配置；这里通常为空。
        self.teacher_agents = teacher_agents  # 保存专家策略；后面用它给每个状态生成 teacher_actions 标签。
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)  # 初始化 skrl Trainer 基类。
        self.teacher_pretrain = self.cfg.get("teacher_pretrain", False)  # 兼容旧配置；真正阈值在 single_agent_train 里读 pretrain_timesteps。
        if self.teacher_pretrain:
            self.pretrain_percentage = 0.2  # 旧逻辑：teacher warm-up 占总训练步数 20%；当前代码没有直接使用这个比例。
        else:
            self.pretrain_percentage = 0.0  # 不使用 teacher warm-up 时比例为 0。
        
        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)
            
    def train(self) -> None:
        """Train the agents with DAgger algorithm

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        self.teacher_agents.set_running_mode("eval")  # teacher 是已经训练好的专家策略，DAgger 阶段只推理，不更新参数。
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")  # student 是被蒸馏的视觉策略，DAgger 阶段需要训练。
            
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_train()
            else:
                raise NotImplementedError("DAgger is not implemented for multi-agent environments")
        else:
            # multi-agent
            raise NotImplementedError("DAgger is not implemented for multi-agent environments")
        
    def eval(self) -> None:
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")
            
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_eval()
            else:
                raise NotImplementedError("DAgger is not implemented for multi-agent environments")
        else:
            # multi-agent
            raise NotImplementedError("DAgger is not implemented for multi-agent environments")
    
    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"
        
        # reset env
        states, infos = self.env.reset()  # states 是环境返回的观测字典，通常同时包含 student 和 teacher 两套观测。
        
        # threshold_timestep = int(self.pretrain_percentage * self.timesteps)
        threshold_timestep = self.cfg.get("pretrain_timesteps", 4000)  # warm-up 阈值：小于该步数时环境由 teacher 动作推进。
        
        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):
            student_obs = states["states"]  # student 看到的输入；在本项目中对应视觉/深度/mask/本体状态等可部署观测。
            teacher_obs = states["obs"]  # teacher 看到的输入；通常是状态型或特权观测，用于生成专家动作标签。
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)  # student 每步交互前的钩子；当前 DAgger agent 里是空实现。
            self.teacher_agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)  # teacher 每步交互前的钩子；保持 skrl 调用流程一致。
            
            with torch.no_grad():  # 采样环境数据时不建立反向传播图；真正训练发生在 agent._update() 中。
                teacher_actions = self.teacher_agents.act(teacher_obs, timestep=timestep, timesteps=self.timesteps)[0]  # teacher 根据特权观测输出专家动作。
                actions = self.agents.act(student_obs, timestep=timestep, timesteps=self.timesteps)[0]  # student 根据视觉观测输出当前策略动作。
                
                if timestep < threshold_timestep:
                    next_states, rewards, terminated, truncated, infos = self.env.step(teacher_actions)  # warm-up：用 teacher 动作推进环境，先收集较稳定的初始数据。
                else:
                    next_states, rewards, terminated, truncated, infos = self.env.step(actions)  # DAgger 主阶段：用 student 动作推进环境，收集 student 自己会遇到的状态分布。
                
                if infos.get("replaced_action", None) is not None: # Use replaced action to learn
                    teacher_actions = infos.get("replaced_action")  # 如果环境内部替换/裁剪了动作，就用替换后的动作作为监督标签。
                    
                # render scene
                if not self.headless:
                    self.env.render()  # 仅用于可视化，不参与 DAgger 数据或 loss 计算。
                
                # record the environments' transitions
                self.agents.record_transition(student_obs=student_obs,
                                              teacher_obs=teacher_obs, # teacher observations：保存 teacher 输入，主要用于日志/兼容 skrl transition。
                                              actions=actions,  # student 当前输出的动作；它不一定被 env.step 使用，warm-up 阶段只是被记录。
                                              teacher_actions=teacher_actions, # teacher actions label：DAgger 的监督学习目标。
                                              rewards=rewards,  # 环境奖励；DAgger loss 本身不用奖励，但 skrl 记录接口需要保留。
                                              next_states=next_states["obs"],  # 下一步 teacher 观测；传给基类记录 transition。
                                              terminated=terminated,  # episode 是否自然结束。
                                              truncated=truncated,  # episode 是否因时间上限等原因截断。
                                              infos=infos,  # 环境附加信息，例如 replaced_action、lifted_now 等。
                                              timestep=timestep,  # 当前训练步数，用于日志和 checkpoint。
                                              timesteps=self.timesteps)  # 总训练步数，用于日志和调度。
                
            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)  # 每步后触发 student；达到 rollouts 间隔时会执行 DAgger 更新。
            
            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():  # reset 不需要梯度。
                    states, infos = self.env.reset()  # 任一并行环境结束时重置环境，开始新 episode。
            else:
                states = next_states  # episode 未结束时，下一轮继续使用刚得到的新状态。
                
    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        states, infos = self.env.reset()
        
        mp4_writers = []
        if self.record_video:
            import imageio, os
            self.env.enable_viewer_sync = False
            for i in range(self.env.num_envs):
                video_name = f"{i}.mp4"
                run_dir = self.cfg["log_dir"]
                path = f"../logs/videos/{run_dir}/{self.cfg['video_name']}"
                if not os.path.exists(path):
                    os.makedirs(path)
                video_name = os.path.join(path, video_name)
                mp4_writer = imageio.get_writer(video_name, fps=10)
                mp4_writers.append(mp4_writer)

        if not self.record_video:
            traj_length = 14000 # int(self.env.max_episode_length)
        else:
            traj_length = int(self.env.max_episode_length)
        
        for timestep in tqdm.tqdm(range(0, traj_length), disable=self.disable_progressbar):

            # if timestep > 20:
            #     break
            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states["states"], timestep=timestep, timesteps=self.timesteps)[0]  # eval 阶段只用 student 控制环境。
                teacher_actions = self.teacher_agents.act(states["obs"], timestep=timestep, timesteps=self.timesteps)[0]  # 这里计算 teacher 动作主要用于对比/调试，未用于 env.step。
                # print("actions: ", actions[1])
                # action_hist.append(actions[1].cpu().numpy())

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)  # 评测时环境由 student 动作推进。

                # render scene
                if not self.headless:
                    self.env.render()
                    
                if self.record_video:
                    imgs = self.env.render_record(mode='rgb_array')
                    if imgs is not None:
                        for i in range(self.env.num_envs):
                            mp4_writers[i].append_data(imgs[i])
                
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    # action_hist = np.array(action_hist)
                    # np.save("action_hist_fix_1.npy", action_hist)
                    states, infos = self.env.reset()
            else:
                states = next_states
                
        if self.record_video:
            for mp4_writer in mp4_writers:
                mp4_writer.close()
    
