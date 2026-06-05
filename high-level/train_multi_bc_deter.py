try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgym.torch_utils import *
except:
    print("Cannot import isaacgym, works only for deployment")

import numpy as np
import torch
import os

import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.utils import set_seed

from modules.feature_extractor import DepthFeatureExtractor
from modules.conv2d import Conv2dHeadModel
from modules.feature_extractor import DepthOnlyFCBackbone58x87, CNNFeatureExtractor, DepthOnlyFCBackbone54x96

from learning.dagger_trainer import DAggerTrainer
from learning.dagger_rnn import DAGGER_DEFAULT_CONFIG as DAGGER_RNN_DEFAULT_CONFIG, DAgger_RNN
from learning.dagger import DAGGER_DEFAULT_CONFIG, DAgger
    
# define models (stochastic and deterministic models) using mixins
class Policy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", use_tanh=False,
                 num_envs=1, num_layers=1, hidden_size=128, sequence_length=16, mode="full", floating_base=False,
                 pitch_control=False, use_roboinfo=True, use_gru=True, deploy=False
                 ):
        Model.__init__(self, observation_space, action_space, device)
        # GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length
        self.mode = mode
        self.use_roboinfo = use_roboinfo
        self.pitch_control = pitch_control
        self.floating_base = floating_base
        self.use_gru = use_gru
        self.deploy = deploy
        
        input_size = 28 + 9 + 64
        if self.pitch_control or self.floating_base:
            input_size += 1
        if (not self.floating_base) and use_roboinfo:
            input_size += 24

        if self.use_gru:
            self.gru = nn.GRU(input_size=input_size, #23 + 9 + 64, #self.num_observations, + self.num_actions + self.num_features
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)
        if not self.use_gru:
            self.mlp = nn.Sequential(nn.Linear(input_size, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.Tanh())
            
        # self.depth_extractor = DepthFeatureExtractor(input_dim=2, output_dim=64)
        if self.mode == "full":
            self.depth_extractor = DepthOnlyFCBackbone54x96(latent_dim=64, output_activation=None, num_channel=12) # TODO: modify the input channel
        elif self.mode == "wrist_seg":
            self.depth_extractor = DepthOnlyFCBackbone54x96(latent_dim=64, output_activation=None, num_channel=9)
        elif self.mode == "front_only":
            self.depth_extractor = DepthOnlyFCBackbone54x96(latent_dim=64, output_activation=None, num_channel=6)
        elif self.mode == "seperate":
            self.depth_extractor_front = DepthOnlyFCBackbone54x96(latent_dim=32, output_activation=None, num_channel=6)
            self.depth_extractor_wrist = DepthOnlyFCBackbone54x96(latent_dim=32, output_activation=None, num_channel=6)

        if use_tanh:
            self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.num_actions),
                                    nn.Tanh())
        else:
            self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.num_actions))
        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        
        if self.use_gru:
            for name, param in self.gru.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states_raw = inputs["states"]
        img_dim = 37
        if self.pitch_control or self.floating_base:
            img_dim += 1    
        if (not self.floating_base) and self.use_roboinfo:
            img_dim += 24
        images = states_raw[:, :-img_dim]
        
        if self.mode == "full":
            images = images.reshape(-1, 12, 54, 96) # TODO: modify the input channel
            if self.deploy:
                images = images[:, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], :, :]
        elif self.mode == "wrist_seg":
            images = images.reshape(-1, 9, 54, 96) # TODO: modify the input channel
            if self.deploy:
                images = images[:, [0, 3, 6, 1, 4, 7, 2, 5, 8], :, :]
        elif self.mode == "front_only":
            images = images.reshape(-1, 6, 54, 96)
            if self.deploy:
                images = images[:, [0, 3, 1, 4, 2, 5], :, :]
        elif self.mode == "seperate":
            images = images.reshape(-1, 12, 54, 96)
            if self.deploy:
                images = images[:, [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11], :, :] # [(forward_mask+forward_depth_seg)*3, (wrist_mask+wrist_depth_seg)*3], each is stacked by 3 frames
            else:
                # images = images[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], :, :] # [forward_mask, wrist_mask, forward_depth_seg, wrist_depth_seg], each is stacked by 3 frames
                images = images[:, [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11], :, :] # [(forward_mask+forward_depth_seg)*3, (wrist_mask+wrist_depth_seg)*3], each is stacked by 3 frames

        # images = images.reshape(-1, 12, 58, 87) # TODO: modify the input channel
        # Sim Order: [forward_mask_1, wrist_mask_1, forward_depth_seg_1, wrist_depth_seg_1,
        #            forward_mask_2, wrist_mask_2, forward_depth_seg_2, wrist_depth_seg_2,
        #           forward_mask_3, wrist_mask_3, forward_depth_seg_3, wrist_depth_seg_3]
        # Real Order: [forward_mask_1, forward_mask_2, forward_mask_3, wrist_mask_1, wrist_mask_2, wrist_mask_3,
        #             forward_depth_seg_1, forward_depth_seg_2, forward_depth_seg_3, wrist_depth_seg_1, wrist_depth_seg_2, wrist_depth_seg_3]
        # images = images[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], :, :] # when 12 channels
        # images = images[:, [0, 4, 8, 1, 5, 9, 2, 6, 10], :, :] # when 9 channels
        # images = images[:, [0, 4, 8, 1, 5, 9], :, :] # when 6 channels
        
        if self.mode == "seperate":
            states = torch.cat([states_raw[:, -img_dim:], self.depth_extractor_front(images[:, :6, ...]), self.depth_extractor_wrist(images[:, 6:, ...])], dim=1)
        else:
            depth_feature = self.depth_extractor(images)
            states = torch.cat([states_raw[:, -img_dim:], depth_feature], dim=1)

        terminated = inputs.get("terminated", None)
        hidden_states = None
        
        if self.use_gru:
            hidden_states = inputs["rnn"][0]
            # training
            if self.training:
                rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
                hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
                # get the hidden states corresponding to the initial sequence
                hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

                # reset the RNN state in the middle of a sequence
                if terminated is not None and torch.any(terminated):
                    rnn_outputs = []
                    terminated = terminated.view(-1, self.sequence_length)
                    indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                    for i in range(len(indexes) - 1):
                        i0, i1 = indexes[i], indexes[i + 1]
                        rnn_output, hidden_states = self.gru(rnn_input[:,i0:i1,:], hidden_states)
                        hidden_states[:, (terminated[:,i1-1]), :] = 0
                        rnn_outputs.append(rnn_output)

                    rnn_output = torch.cat(rnn_outputs, dim=1)
                # no need to reset the RNN state in the sequence
                else:
                    rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
            # rollout
            else:
                rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)

            # flatten the RNN output
            prev_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)
            # return self.net(rnn_output), self.log_std_parameter, {"rnn": [hidden_states]}
            actions = self.net(prev_output)
            # actions[:, 6] = torch.sigmoid(actions[:, 6]) # TODO: this is sigmoid for gripper
            return actions, {"rnn": [hidden_states]}
    
        else:
            prev_output = self.mlp(states)
            actions = self.net(prev_output)
            return actions, {}
    
    
def create_env(cfg, args, mode):
    from envs import B1Z1PickMulti, B1Z1Float
    import utils.wrapper as wrapper

    cfg["sensor"]["enableCamera"] = True
    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["cameraMode"] = mode
    if args.last_commands:
        cfg["env"]["lastCommands"] = True
    if args.record_video:
        cfg["record_video"] = True
    if args.control_freq is not None:
        cfg["env"]["controlFrequencyLow"] = int(args.control_freq)
    robot_start_pose = (-2.00, 0, 0.55)
    if args.eval:
        robot_start_pose = (-1.5, 0, 0.55)
    _env = eval(args.task)(cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device, 
                         graphics_device_id=args.graphics_device_id, headless=args.headless, 
                         use_roboinfo=args.roboinfo, observe_gait_commands=args.observe_gait_commands, robot_start_pose=robot_start_pose,
                         no_feature=args.no_feature, mask_arm=args.mask_arm, depth_random=args.depth_random, stu_distill=True, pitch_control=args.pitch_control, pred_success=args.pred_success,
                         rand_control=args.rand_control, arm_delay=args.arm_delay, rand_cmd_scale=args.rand_cmd_scale, rand_depth_clip=args.rand_depth_clip, stop_pick=args.stop_pick, arm_kp=args.arm_kp, arm_kd=args.arm_kd, table_height=args.table_height, eval=args.eval)
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env

def get_trainer(is_eval=False):
    from utils.config import load_cfg, get_params, copy_cfg
    
    args = get_params()
    set_seed(args.seed)
    args.eval = is_eval
    use_roboinfo = args.roboinfo
    if args.wrist_seg:
        mode = "wrist_seg"
    elif args.front_only:
        mode = "front_only"
    elif args.seperate:
        mode = "seperate"
    else:
        mode = "full"
    
    # assert use_roboinfo, "Are you sure not using roboinfo?" # TODO: temporarily for reminder
    args.wandb = args.wandb and (not args.eval) and (not args.debug)
    
    cfg_file = "b1z1_" + args.task[4:].lower() + ".yaml"
    file_path = "data/cfg/" + cfg_file
    
    if args.resume:
        experiment_dir = os.path.join(args.experiment_dir, args.wandb_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        pt_files = os.listdir(checkpoint_dir)
        pt_files = [file for file in pt_files if file.endswith(".pt") and (not file.startswith("best"))]
        # Find the latest checkpoint
        checkpoint_steps = 0
        if len(pt_files) > 0:
            args.checkpoint = os.path.join(checkpoint_dir, sorted(pt_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1])
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
        cfg_files = os.listdir(experiment_dir)
        cfg_files = [file for file in cfg_files if file.endswith(".yaml")]
        if len(cfg_files) > 0:
            cfg_file = cfg_files[0]
            file_path = os.path.join(experiment_dir, cfg_file)
        
        print("Find the latest checkpoint: ", args.checkpoint)
        
    cfg = load_cfg(file_path)
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = args.use_tanh
    cfg['env']["near_goal_stop"] = args.near_goal_stop
    cfg['env']["obj_move_prob"] = args.obj_move_prob
    if args.eval or args.debug:
        cfg['env']['numEnvs'] = 34
        if args.checkpoint:
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
            cfg["env"]["globalStepCounter"] = checkpoint_steps
    env = create_env(cfg=cfg, args=args, mode=mode)
    device = env.rl_device
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)  # DAgger 聚合数据的短期缓冲区；rollouts=24 时每 24 步更新一次。

    student_action_space = env.action_space  # student 最终要输出和 teacher 相同维度的高层动作。
    student_obs_space = env.observation_space  # wrapper 暴露给 student policy 的观测空间。
    if args.pred_success:
        student_action_space = (student_action_space.shape[0]+1,)  # pred_success 会让 student 多输出 1 维 lifted/success 预测。
    
    model_dagger = {}  # skrl agent 需要用字典传入模型。
    model_dagger["policy"] = Policy(student_obs_space, student_action_space, device, num_envs=env.num_envs, mode=mode, use_roboinfo=use_roboinfo, use_tanh=args.use_tanh, use_gru=not args.mlp_stu, pitch_control=args.pitch_control, floating_base=cfg["env"].get("floatingBase", False))  # student 视觉策略；默认 use_gru=True，除非命令行加 --mlp_stu。
    
    dagger_config = DAGGER_DEFAULT_CONFIG if args.mlp_stu else DAGGER_RNN_DEFAULT_CONFIG  # MLP student 用 DAgger，GRU student 用 DAgger_RNN。
    
    cfg_dagger = dagger_config.copy()  # 复制默认 DAgger 配置，下面按当前实验覆盖关键超参。
    cfg_dagger["rollouts"] = 24  # memory_size；每收集 24 个 step 的聚合数据后触发一次 student 更新。
    cfg_dagger["learning_epochs"] = 5  # 每次更新重复遍历 memory 数据 5 轮。
    cfg_dagger["mini_batches"] = 3  # 24 * 8192 / 32768；把一次收集的数据分成 3 个 mini-batch。
    cfg_dagger["discount_factor"] = 0.99  # 兼容 skrl/PPO 字段；当前 DAgger MSE 不直接使用。
    cfg_dagger["lambda"] = 0.95  # 兼容 skrl/PPO 字段；当前 DAgger MSE 不直接使用。
    cfg_dagger["learning_rate"] = 5e-5  # student policy 的 Adam 学习率。
    cfg_dagger["learning_rate_scheduler"] = None  # DAgger 训练中不启用学习率调度。
    # cfg_dagger["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_dagger["random_timesteps"] = 0  # 不使用随机动作阶段；student 从第一步就输出自己的动作。
    cfg_dagger["learning_starts"] = 0  # 不额外延迟更新；memory 满一个 rollouts 后即可训练。
    cfg_dagger["grad_norm_clip"] = 1.0  # student 反向传播时的梯度裁剪阈值。
    cfg_dagger["ratio_clip"] = 0.2  # 兼容 skrl/PPO 字段；当前 DAgger loss 不直接使用。
    cfg_dagger["value_clip"] = 0.2  # 兼容 skrl/PPO 字段；当前没有 value loss。
    cfg_dagger["clip_predicted_values"] = True  # 兼容 skrl/PPO 字段；当前没有 value prediction。
    cfg_dagger["value_loss_scale"] = 1.0  # 兼容 skrl/PPO 字段；当前没有 value loss。
    cfg_dagger["kl_threshold"] = 0  # KL 早停关闭；当前 DAgger 更新中 KL 计算也未启用。
    cfg_dagger["rewards_shaper"] = None  # 不对 reward 做 shaping；DAgger 核心 loss 是动作 MSE。
    # logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
    cfg_dagger["experiment"]["write_interval"] = 24  # 每 24 步写一次日志。
    cfg_dagger["experiment"]["checkpoint_interval"] = 1000  # 每 1000 步保存一次 student checkpoint。
    cfg_dagger["experiment"]["directory"] = args.experiment_dir  # checkpoint/log 的父目录。
    cfg_dagger["experiment"]["experiment_name"] = args.wandb_name  # 实验名，也用于日志目录名。
    cfg_dagger["experiment"]["wandb"] = args.wandb  # 是否同步到 W&B。
    # Train a fixed base policy
    cfg_dagger["fixed_base"] = args.fixed_base  # 传给 DAgger loss：固定底盘时忽略最后 2 个动作维度。
    cfg_dagger["reach_only"] = args.reach_only  # 传给 DAgger loss：只训练 reach 相关动作维度。
    cfg_dagger["pred_success"] = args.pred_success  # 传给 DAgger loss：额外监督 lifted/success 预测。
    if args.wandb:
        cfg_dagger["experiment"]["wandb_kwargs"] = {"project": args.wandb_project, "tensorboard": False, "name": args.wandb_name}  # W&B 项目名和 run 名配置。
    
    if args.mlp_stu:
        agent = DAgger(models=model_dagger,
                    memory=memory,  # 保存 student_obs、teacher_actions 等 DAgger 训练样本。
                    cfg=cfg_dagger,  # 上面配置好的 DAgger 超参。
                    observation_space=student_obs_space,  # teacher_obs 记录空间；skrl agent 的 observation_space 字段。
                    action_space=env.action_space,  # teacher/student 对齐的动作空间。
                    state_space=env.state_space,  # student_obs 空间；DAgger.update 时实际喂给 student policy。
                    device=device)  # 训练设备，通常是 cuda。
    else:
        agent = DAgger_RNN(models=model_dagger,
                memory=memory,  # 保存 student_obs、teacher_actions，以及 GRU hidden state。
                cfg=cfg_dagger,  # 上面配置好的 RNN DAgger 超参。
                observation_space=student_obs_space,  # teacher_obs 记录空间；skrl agent 的 observation_space 字段。
                action_space=env.action_space,  # teacher/student 对齐的动作空间。
                state_space=env.state_space,  # student_obs 空间；DAgger_RNN.update 时实际喂给 student policy。
                device=device)  # 训练设备，通常是 cuda。
    
    from train_multistate import Policy as PPOPolicy
    from train_multistate import Value as PPOValue
    from train_multistate import PPO_DEFAULT_CONFIG, PPO

    teacher_action_space = env.action_space  # teacher 输出的动作空间，必须和 student 要模仿的动作空间一致。
    teacher_obs_space = env.observation_space  # teacher 输入空间；论文里对应 privileged/state-based policy 的观测。
    # ------------------- PPO CONFIG -------------------
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()  # teacher 是 PPO 类实例，但在 DAgger 里只加载 checkpoint 并 eval 推理。
    cfg_ppo["rollouts"] = 24  # memory_size；这里主要为了构造 PPO agent，DAgger 阶段不训练 teacher。
    cfg_ppo["learning_epochs"] = 5  # teacher 训练超参；DAgger 阶段不会实际更新 teacher。
    cfg_ppo["mini_batches"] = 6  # 24 * 8192 / 32768；teacher 训练超参，DAgger 阶段只保留配置。
    cfg_ppo["discount_factor"] = 0.99  # teacher PPO 配置字段。
    cfg_ppo["lambda"] = 0.95  # teacher PPO 配置字段。
    cfg_ppo["learning_rate"] = 5e-4  # teacher PPO 配置字段；DAgger 阶段不优化 teacher。
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL  # teacher PPO 配置字段；DAgger 阶段不优化 teacher。
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}  # teacher PPO 配置字段。
    cfg_ppo["random_timesteps"] = 0  # teacher PPO 配置字段。
    cfg_ppo["learning_starts"] = 0  # teacher PPO 配置字段。
    cfg_ppo["grad_norm_clip"] = 1.0  # teacher PPO 配置字段。
    cfg_ppo["ratio_clip"] = 0.2  # teacher PPO 配置字段。
    cfg_ppo["value_clip"] = 0.2  # teacher PPO 配置字段。
    cfg_ppo["clip_predicted_values"] = True  # teacher PPO 配置字段。
    cfg_ppo["value_loss_scale"] = 1.0  # teacher PPO 配置字段。
    cfg_ppo["kl_threshold"] = 0  # teacher PPO 配置字段。
    cfg_ppo["rewards_shaper"] = None  # teacher PPO 配置字段。
    cfg_ppo["state_preprocessor"] = RunningStandardScaler  # 加载 teacher checkpoint 时需要同样的观测归一化模块。
    cfg_ppo["state_preprocessor_kwargs"] = {"size": teacher_obs_space, "device": device}  # teacher 观测归一化器尺寸和设备。
    cfg_ppo["value_preprocessor"] = RunningStandardScaler  # 加载 teacher checkpoint 时需要同样的 value 归一化模块。
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}  # value 归一化器尺寸和设备。

    ppo_models = {}  # skrl PPO agent 需要 policy/value 两个模型。
    ppo_models["policy"] = PPOPolicy(teacher_obs_space, teacher_action_space, device, num_features=1024, encode_dim=128, clip_actions=args.use_tanh, deterministic=True)  # teacher policy；deterministic=True 表示生成稳定专家标签。
    ppo_models["value"] = PPOValue(teacher_obs_space, teacher_action_space, device, num_features=1024, encode_dim=128)  # teacher value；DAgger 推理时不用它训练，但 PPO agent 构造需要。
    ppo_agent = PPO(models=ppo_models,
                memory=memory,  # 与 student 共用 memory 对象；DAggerTrainer 只用 teacher act，不让 teacher 记录训练样本。
                cfg=cfg_ppo,  # teacher PPO agent 的配置。
                observation_space=teacher_obs_space,  # teacher 输入空间。
                action_space=teacher_action_space,  # teacher 输出空间。
                device=device)  # teacher 推理设备。
        
    if not args.eval:
        print("load teacher ckpt: ", args.teacher_ckpt_path)
        ppo_agent.load(args.teacher_ckpt_path)  # 加载已经训练好的 teacher；DAgger 用它在线给 student 生成动作标签。
    
    # cfg_trainer = {"timesteps": args.timesteps, "headless": True, "teacher_pretrain": True}
    cfg_trainer = {"timesteps": args.timesteps, "teacher_pretrain": True}  # DAggerTrainer 配置；真正 warm-up 步数由下一行控制。
    cfg_trainer["pretrain_timesteps"] = 8000 if args.depth_random else 4000  # 前若干步用 teacher 动作推进环境，对应论文中的 teacher sampling warm-up。
    if args.checkpoint:
        print("Resuming from checkpoint: ", args.checkpoint)
        agent.load(args.checkpoint)  # 恢复 student checkpoint，继续 DAgger 训练或评测。
        checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])  # 从文件名解析已训练步数，例如 model_1000.pt -> 1000。
        if args.record_video:
            experiment_dir = args.checkpoint.split("/")[0]  # 记录视频时从 checkpoint 路径解析实验目录。
            wandb_name = args.checkpoint.split("/")[1]  # 记录视频时从 checkpoint 路径解析 run 名。
            cfg_trainer["video_name"] = wandb_name +"-"+str(checkpoint_steps)  # 视频目录/文件名包含 run 名和步数。
            cfg_trainer["log_dir"] = experiment_dir  # 视频保存时使用的日志目录名。
            cfg_trainer["record_video"] = True  # 打开 DAggerTrainer.eval 里的视频保存逻辑。
        if not args.eval:
            cfg_trainer["initial_timestep"] = checkpoint_steps  # 续训时从 checkpoint 对应步数继续计数。
        
    trainer = DAggerTrainer(cfg=cfg_trainer, env=env, agents=agent, teacher_agents=ppo_agent)  # 把 student agent 和 teacher PPO 交给 DAggerTrainer 统一调度。
    if args.wandb:
        import wandb
        wandb.save("data/cfg/" + cfg_file, policy="now")
        wandb.save("envs/b1z1_" + args.task[4:].lower() + ".py", policy="now")
        wandb.save("train_multi_bc_deter_full.py", policy="now")
    if not args.eval:
        if not os.path.exists(os.path.join(args.experiment_dir, args.wandb_name, cfg_file)):
            copy_cfg(file_path, os.path.join(args.experiment_dir, args.wandb_name))
    return trainer
    
    
if __name__ == "__main__":
    trainer = get_trainer()
    trainer.train()
