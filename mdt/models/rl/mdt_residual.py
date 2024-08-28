from pathlib import Path
from typing import List, Type, Dict
from collections import deque

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        net_arch: List,
        init_log_std: float,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.net_arch = net_arch
        self.network = create_mlp(self.observation_dim, self.action_dim, self.net_arch)
        self.log_std = nn.Parameter(torch.ones(1, self.action_dim) * init_log_std)

    def get_action(self, norm_obs: torch.Tensor, deterministic: bool = False):
        action_mean = self.network(norm_obs)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        distribution = Normal(action_mean, action_std)
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.rsample()

        return (
            action,
            distribution.log_prob(action).sum(dim=1),
            distribution.entropy().sum(dim=1),
        )


class Critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.latent_dim = 1
        self.net_arch = net_arch
        self.network = create_mlp(self.observation_dim, 1, self.net_arch, activation_fn)

    def forward(self, norm_obs: torch.Tensor):
        return self.network(norm_obs)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    layers = []
    layers.append(nn.Linear(input_dim, net_arch[0]))
    layers.append(activation_fn())
    for i in range(1, len(net_arch)):
        layers.append(nn.Linear(net_arch[i - 1], net_arch[i]))
        layers.append(activation_fn())
    layers.append(nn.Linear(net_arch[-1], output_dim))
    return nn.Sequential(*layers)


class ResidualPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_net_arch: List[int],
        critic_net_arch: List[int],
        init_log_std: float = 0,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.actor = Actor(
            observation_dim, action_dim, actor_net_arch, init_log_std, activation_fn
        )
        self.critic = Critic(observation_dim, critic_net_arch, activation_fn)

    def get_action(self, norm_obs, deterministic):
        return self.actor.get_action(norm_obs, deterministic)

    def get_value(self, norm_obs):
        return self.critic(norm_obs)



class MDTResidual(nn.Module):
    def __init__(
        self,
        mdt_config: DictConfig,
        residual_config: DictConfig,
        tactile_encoder_config: DictConfig,
        visual_encoder_config: DictConfig,
        device: str = "cuda",
        seed: int = 42,
    ):
        super().__init__()
        self.mdt_config = mdt_config
        self.device = device
        self.seed = seed
        self.residual_policy = hydra.utils.instantiate(residual_config).to(self.device)
        self.tactile_encoder = hydra.utils.instantiate(tactile_encoder_config).to(self.device)
        self.visual_encoder = hydra.utils.instantiate(visual_encoder_config).to(self.device)
        self.base_actions = deque(maxlen=mdt_config.multistep)
        self.action_horizon = mdt_config.multistep


    def configure_mdt(self):
        self.mdt = hydra.utils.instantiate(self.mdt_config).to(self.device)
        mdt_ckpt_path = Path(self.mdt_config.mdt_ckpt_path)
        if not mdt_ckpt_path.exists():
            raise ValueError(f"mdt checkpoint path: {mdt_ckpt_path} is not valid")
        self.mdt.load_pretrained_parameters(mdt_ckpt_path)


    def sample_action(self, obs, goal):
        if not self.base_actions:
            actions = self.sample_base_action(obs, goal)
            for i in range(self.action_horizon):
                self.base_actions.append(actions[:, i, :])



    @torch.no_grad()
    def sample_base_action(self, obs, goal):
        actions = self.mdt(obs, goal)
        return actions

