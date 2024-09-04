from copy import deepcopy
from tqdm.rich import tqdm
import numpy as np
import torch
import time
from collections import deque
from torch.nn import functional as F
from typing import Dict, List, Optional, Any, Union
import wandb
import sys

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
        stats_window_size: int = 100,
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
        self.rollout_buffer = RolloutBuffer(
            buffer_size,
            observation_dim,
            action_dim,
            device,
            gae_lambda,
            gamma,
            env.num_envs,
        )
        self._n_updates = 0
        self.ep_info_buffer = None
        self.ep_success_buffer = None
        self.stats_window_size = stats_window_size

    def collect_rollout(self, n_rollout_steps):
        self.policy.eval()
        n_steps = 0
        self.rollout_buffer.reset()
        while n_steps < n_rollout_steps:
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                actions, residual_actions, values, log_probs, last_residual_obs = (
                    self.policy(self._last_obs)
                )
            new_obs, rewards, dones, infos = self.env.step(actions)

            self.num_timesteps += self.env.num_envs

            # Give access to local variables

            self._update_info_buffer(infos, dones)
            n_steps += 1


            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = infos[idx]["terminal_observation"]

                    # estimate terminal value for env[idx]
                    with torch.no_grad():
                        terminal_value = self.policy.get_values_for_env(terminal_obs, idx)  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            self.rollout_buffer.add(
                last_residual_obs.cpu(),
                residual_actions.cpu(),
                rewards.cpu(),
                self._last_episode_starts.cpu(),  # type: ignore[arg-type]
                values.cpu(),
                log_probs.cpu(),
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
                    

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.get_values(deepcopy(new_obs))  # type: ignore[arg-type]

        self.rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )
        return True

    def train_policy(self) -> None:
        self.policy.train()
        entropy_losses = []
        pg_losses, value_losses = [], []
        losses = []
        clip_fractions = []
        clip_range = self.clip_range

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()
                ).item()
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

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                losses.append(loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
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
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
                if self.policy.lr_scheduler is not None:
                    self.policy.lr_scheduler.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )


        wandb.log({"train/entropy_loss": np.mean(entropy_losses)})
        wandb.log({"train/policy_gradient_loss": np.mean(pg_losses)})
        wandb.log({"train/value_loss": np.mean(value_losses)})
        wandb.log({"train/approx_kl": np.mean(approx_kl_divs)})
        wandb.log({"train/clip_fraction": np.mean(clip_fractions)})
        wandb.log({"train/loss": np.mean(losses)})
        wandb.log({"train/explained_variance": explained_var})
        if hasattr(self.policy, "log_std"):
            wandb.log({"train/std": torch.exp(self.policy.log_std).mean().item()})

        wandb.log({"train/n_updates": self._n_updates})
        wandb.log({"train/clip_range": clip_range})

    def learn(self, total_timesteps: int, log_interval: int):
        total_timesteps = self._setup_learn(total_timesteps)
        iteration = 0
        pbar = tqdm(total=total_timesteps-self.num_timesteps)
        while self.num_timesteps < total_timesteps:
            start_timesteps = self.num_timesteps
            continue_learning = self.collect_rollout(self.n_steps)
            end_timesteps = self.num_timesteps
            pbar.update(end_timesteps-start_timesteps)
            if not continue_learning:
                break
            iteration +=1
            if iteration % log_interval == 0:
                self._dump_logs(iteration)
            self.train_policy()
        pbar.refresh()
        pbar.close()

    def _setup_learn(self, total_timesteps) -> int:
        self.start_time = time.time_ns()
        if self.ep_info_buffer is None:
            self.ep_info_buffer = deque(maxlen=self.stats_window_size)
            self.ep_success_buffer = deque(maxlen=self.stats_window_size)

        total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps
        if self._last_obs is None:
            self._last_obs, _ = self.env.reset()
            self._last_episode_starts = torch.ones((self.env.num_envs,), dtype=torch.bool)

        return total_timesteps

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        wandb.log({"time/iterations": iteration})
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            wandb.log({"rollout/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])})
            wandb.log({"rollout/ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])})
        wandb.log({"time/fps": fps})
        wandb.log({"time/time_elapsed": int(time_elapsed)})
        wandb.log({"time/total_timesteps": self.num_timesteps})
        if len(self.ep_success_buffer) > 0:
            wandb.log({"rollout/success_rate": safe_mean(self.ep_success_buffer)})

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[torch.Tensor] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = torch.Tensor([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
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
    assert len(y_true.shape) == 1 and len(y_pred.shape) == 1
    var_y = torch.var(y_true)
    return torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y  # type: ignore

def safe_mean(arr: Union[np.ndarray, torch.Tensor, list, deque]) -> float:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    if isinstance(arr, torch.Tensor):
        arr.numpy()
    return np.nan if len(arr) == 0 else float(np.mean(arr))  # type: ignore[arg-type]

