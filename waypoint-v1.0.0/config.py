from dataclasses import dataclass
from datetime import datetime
import os
import random
import sys
import re
from pathlib import Path


@dataclass
class Env_Args:
    ip: str = '127.0.0.1'
    port: int = 8000 + os.getpid() % 1000 + random.randint(1, 999)  # 确保端口不冲突
    play_mode: int = 0
    red_num: int = 1
    blue_num: int = 0
    red_com: int = 0
    blue_com: int = 0
    scenes: int = 3  # 场景三没有地面高山等障碍，防止飞机意外，用于训练
    render: int = 0
    action_dim: int = 4
    obs_dim: int = 15
    control_side: str = 'red'
    excute_path: str = r'/root/nolinux/ZK.x86_64' if sys.platform == "linux" else rf'{os.environ.get("USERPROFILE")}\Desktop\MM\windows\ZK.exe'
    obs_scale: bool = True
    """whether to scale the observation"""


@dataclass
class Run_Args:
    wandb_notes: str = "fix helper and 2.0.3 remove heading reward change h reward"
    """the notes of this run"""
    version: str = re.search(r'v\d+\.\d+\.\d+', Path(sys.argv[0]).resolve().parent.name).group()
    """the version of this project"""
    wandb_project_name: str = f"Jsbsim-navi-{version}"
    """the wandb's project name"""
    vec_normalize: bool = False
    """whether to normalize the observation"""
    sbx: bool = True
    """whether to use sbx"""
    run_name: str = datetime.now().strftime("%m_%d_%H_%M") + "_time"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_freq: int = 2048000
    """the frequency of saving checkpoints and """
    wandb_entity: str = "oterea-group"
    """the entity (team) of wandb's project"""
    wandb_apikey: str = "2c7bd7fcfa4b6e700f733a5bb9cbb36701b2d14c"
    """the apikey of wandb"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    # ==========================save checkpoint args============================
    model_path: str = "checkpoints"
    """where to save the checkpoints"""

    # Algorithm specific arguments
    env_id: str = "zk-v0.02"
    """the id of the environment"""
    total_timesteps: int = 20000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    # target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
