from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Type
import pytorch_lightning as pl

import einops
import hydra
import torch
import torchvision
import torchvision.models as models
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Normal

from mdt.models.mdtv_agent import MDTVAgent
from mdt.models.rl.utils import TASK_TO_LABEL


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
        self.network = create_mlp(
            self.observation_dim, self.action_dim, self.net_arch, activation_fn
        )
        self.log_std = nn.Parameter(torch.ones(1, self.action_dim) * init_log_std)

    def get_actions_and_log_probs(
        self, norm_obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple:
        distribution = self.get_distribution(norm_obs)
        if deterministic:
            actions = distribution.mean
        else:
            actions = distribution.rsample()

        return (
            actions,
            distribution.log_prob(actions).sum(dim=-1),
        )

    def get_log_probs_and_entropy(self, norm_obs, actions):
        distribution = self.get_distribution(norm_obs)
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return (log_probs, entropy)

    def get_distribution(self, norm_obs):
        action_mean = self.network(norm_obs)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        distribution = Normal(action_mean, action_std)
        return distribution


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


class ResidualPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        actor_net_arch: List[int],
        critic_net_arch: List[int],
        init_log_std: float = 0,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: str="cuda",
    ):
        super().__init__()
        self.device = device
        self.actor = Actor(
            observation_dim, action_dim, actor_net_arch, init_log_std, activation_fn
        )
        self.critic = Critic(observation_dim, critic_net_arch, activation_fn)

    def get_actions_and_values(
        self, norm_obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_probs = self.actor.get_actions_and_log_probs(
            norm_obs, deterministic
        )
        values = self.critic(norm_obs)
        return actions, values, log_probs

    def evaluate_actions(
        self, norm_obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropy = self.actor.get_log_probs_and_entropy(norm_obs, actions)
        values = self.critic(norm_obs)
        return values, log_probs, entropy

    def get_actions(self, norm_obs: torch.Tensor, deterministic=False) -> torch.Tensor:
        return self.actor.get_actions_and_log_probs(norm_obs, deterministic)[0]

    def get_values(self, norm_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(norm_obs)


class MDTVResidual(nn.Module):
    def __init__(
        self,
        mdtv: DictConfig,
        residual: DictConfig,
        transforms: DictConfig,
        encoder: DictConfig,
        optimizer: DictConfig,
        device: str = "cuda",
        seed: int = 42,
        action_scale: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.task2label = TASK_TO_LABEL
        self.label2task = {value: key for key, value in self.task2label.items()}
        self.mdtv_config = mdtv
        self.residual_config = residual
        self.transforms_config = transforms
        self.optimizer_config = optimizer

        # self.device = device
        self.seed = seed
        self.residual_policy: ResidualPolicy = hydra.utils.instantiate(self.residual_config).to(
            self.device
        )
        self.residual_action_scale = action_scale
        self.encoder_config = encoder
        self.tactile_encoder: ResNet = hydra.utils.instantiate(
            self.encoder_config.tactile_encoder
        ).to(self.device)
        self.static_encoder: ResNet = hydra.utils.instantiate(
            self.encoder_config.visual_encoder
        ).to(self.device)
        self.gripper_encoder: ResNet = hydra.utils.instantiate(
            self.encoder_config.visual_encoder
        ).to(self.device)
        self.base_actions = deque(maxlen=mdtv.multistep)
        self.action_horizon = mdtv.multistep
        self.optimizer, self.lr_scheduler = self.configure_optimizer()
        self._setup_transforms()

    def configure_mdtv(self) -> None:
        self.mdtv: MDTVAgent = hydra.utils.instantiate(self.mdtv_config)
        self.mdtv.to(self.device)
        # mdtv_ckpt_path = Path(self.mdtv_config.mdtv_ckpt_path)
        # if not mdtv_ckpt_path.exists():
        #     raise ValueError(f"mdtv checkpoint path: {mdtv_ckpt_path} is not valid")
        # self.mdtv.load_pretrained_parameters(mdtv_ckpt_path)

    def configure_optimizer(
        self,
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optim_groups = []
        optim_groups.extend(
            [
                {
                    "params": self.tactile_encoder.parameters(),
                    "lr": self.optimizer_config.tactile_encoder_lr,
                }
            ]
        )
        optim_groups.extend(
            [
                {
                    "params": self.static_encoder.parameters(),
                    "lr": self.optimizer_config.visual_encoder_lr,
                }
            ]
        )
        optim_groups.extend(
            [
                {
                    "params": self.gripper_encoder.parameters(),
                    "lr": self.optimizer_config.visual_encoder_lr,
                }
            ]
        )
        optim_groups.extend(
            [
                {
                    "params": self.residual_policy.actor.parameters(),
                    "lr": self.optimizer_config.actor_lr,
                }
            ]
        )
        optim_groups.extend(
            [
                {
                    "params": self.residual_policy.critic.parameters(),
                    "lr": self.optimizer_config.critic_lr,
                }
            ]
        )
        optimizer = torch.optim.Adam(optim_groups)
        return optimizer, None

    @torch.no_grad()
    def sample_base_actions(self, raw_obs: Dict) -> torch.Tensor:
        goal_labels = raw_obs["goal_label"].to(torch.int)
        goal_emb = raw_obs["goal_emb"]
        goal_annotation = [self.label2task[label.item()] for label in goal_labels]
        goal = {
            "lang_text": goal_annotation,
            "lang": goal_emb,
        }
        raw_obs = self.process_mdtv_obs(raw_obs)
        actions = self.mdtv(raw_obs, goal)
        return actions

    def forward(
        self, raw_obs: Dict
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not self.base_actions:
            actions = self.sample_base_actions(raw_obs)
            for i in range(self.action_horizon):
                self.base_actions.append(actions[:, i, :])
        base_actions = self.base_actions.popleft()
        residual_obs = self.process_residual_obs(raw_obs, base_actions)
        residual_actions, values, log_probs = (
            self.residual_policy.get_actions_and_values(residual_obs)
        )
        actions = base_actions + residual_actions * self.residual_action_scale
        actions = torch.clip(actions, -1, 1)
        return actions, residual_actions, values, log_probs, residual_obs

    def get_values(self, raw_obs):
        if not self.base_actions:
            base_actions = self.sample_base_actions(raw_obs)
            next_base_action = base_actions[:, 0, :]
        else:
            next_base_action = self.base_actions[0]
        residual_obs = self.process_residual_obs(raw_obs, next_base_action)
        return self.residual_policy.get_values(residual_obs)

    def get_values_for_env(self, raw_obs, idx):
        if not self.base_actions:
            base_actions = self.sample_base_actions(raw_obs)
            next_base_action = base_actions[:, 0, :]
        else:
            next_base_action = self.base_actions[0][idx].unsueeze(0)
        residual_obs = self.process_residual_obs(raw_obs, next_base_action)
        return self.residual_policy.get_values(residual_obs)
    def evaluate_actions(self, residual_obs, action):
        return self.residual_policy.evaluate_actions(residual_obs, action)

    def process_residual_obs(
        self, raw_obs: Dict[str, torch.Tensor], base_actions: torch.Tensor
    ) -> torch.Tensor:
        goal_emb = raw_obs["goal_emb"].to(self.residual_policy.device)
        # assert len(base_actions.shape) == 2, f"base_actions has shape {base_actions.shape}, expected (b, 7)"
        del raw_obs['goal_label']
        del raw_obs['goal_emb']
        for key, obs in raw_obs.items():
            raw_obs[key] = self.residual_transforms[key](obs)
        tactile_image = raw_obs["rgb_tactile"].to(self.residual_policy.device)

        tactile_image = einops.rearrange(
            tactile_image, "b (n c) w h -> (b n) c w h", n=2, c=3
        )
        static_image = raw_obs["rgb_static"].to(self.residual_policy.device)
        gripper_image = raw_obs["rgb_static"].to(self.residual_policy.device)

        tactile_emb = einops.rearrange(
            self.tactile_encoder(tactile_image), "(b n) d -> b (n d)", n=2
        )
        static_emb = self.static_encoder(static_image)
        gripper_emb = self.gripper_encoder(gripper_image)
        # assert len(static_emb.shape) == 2, f"static_emb has shape {static_emb.shape}, expected (b, d)"
        # assert len(gripper_emb.shape) == 2, f"static_emb has shape {gripper_emb.shape}, expected (b, d)"
        residual_obs = torch.cat(
            [static_emb, gripper_emb, tactile_emb, base_actions, goal_emb], dim=-1
        ).to(self.residual_policy.device)

        return residual_obs

    def process_mdtv_obs(
        self, raw_obs: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return_obs = {}
        return_obs["rgb_obs"] = {}
        return_obs["depth_obs"] = {}
        for key, obs in raw_obs.items():
            if "rgb" in key:
                return_obs["rgb_obs"][key] = self.mdtv_transforms[key](obs).unsqueeze(1).to(
                    self.mdtv.device
                )
            elif "depth" in key:
                return_obs["depth_obs"][key] = self.mdtv_transforms[key](obs).unsqueeze(1).to(
                    self.mdtv.device
                )
            else:
                continue

        return return_obs

    def _setup_transforms(self):
        self.mdtv_transforms = {}
        self.residual_transforms = {}
        for obs in self.transforms_config.mdtv:
            obs_transforms = []
            for transform in self.transforms_config.mdtv[obs]:
                instantiate_transform = hydra.utils.instantiate(
                    transform, _convert_="all"
                )
                obs_transforms.append(instantiate_transform)
            self.mdtv_transforms[obs] = torchvision.transforms.v2.Compose(
                obs_transforms
            )

        for obs in self.transforms_config.residual:
            obs_transforms = []
            for transform in self.transforms_config.residual[obs]:
                instantiate_transform = hydra.utils.instantiate(
                    transform, _convert_="all"
                )
                obs_transforms.append(instantiate_transform)
            self.residual_transforms[obs] = torchvision.transforms.v2.Compose(
                obs_transforms
            )

    def reset(self):
        self.base_actions = deque(maxlen=self.mdtv_config.multistep)

class ResNet(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_groups: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        resnet = models.resnet18(pretrained=pretrained)
        n_inputs = resnet.fc.in_features
        modules = list(resnet.children())[:-1]
        self.model = nn.Sequential(*modules)
        if freeze_backbone:
            freeze_model(self.model)

        replace_submodules(
            root_module=self.model,
            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            func=lambda x: nn.GroupNorm(
                num_groups=x.num_features // num_groups, num_channels=x.num_features
            ),
        )
        self.proj = nn.Linear(n_inputs, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        time_series = False

        assert len(x.shape) > 3
        if len(x.shape) == 5:
            x = einops.rearrange(x, "b t c w h -> (b t) c w h")
            time_series = True

        x = self.model(x).flatten(start_dim=1)
        x = self.proj(x)
        if time_series:
            x = einops.rearrange(x, "(b t) d -> b t d", b=batch_size)

        return x


def freeze_model(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


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
