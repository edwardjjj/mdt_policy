import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
from gym import spaces

from .buffer import RolloutBuffer


class PPO:
    def __init__(
        self,
        env,
        observation_dim,
        action_dim,
        buffer_size,
        policy,
        device,
        n_steps: int,
        batch_size: int,
        n_epochs: int,
        gamma: float,
        gae_lambda: float,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        seed: int = 42,
    ):

        self.num_timesteps = 0
        self._total_timesteps = 0
        self._num_timesteps_at_start = 0
        self.start_time = 0.0
        self._last_obs = None
        self._last_episode_starts = None
        self._episode_num = 0
        self.env = env
        self.policy = policy
        self.device = device
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.rollout_buffer = RolloutBuffer(buffer_size, observation_dim, action_dim, device, gae_lambda, gamma, env.n_envs)
        self._n_updates = 0

    def train(self) -> None:
        self.policy.set_training_mode(True)
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        clip_range = self.clip_range

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                
                values_pred = values
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                # if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                #     continue_training = False
                #     if self.verbose >= 1:
                #         print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                #     break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y # type: ignore
