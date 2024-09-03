from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from gym import spaces


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        device: Union[torch.device, str] = "cuda",
        num_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
        self.device = device
        self.num_envs = num_envs
    @staticmethod
    def swap_and_flatten(arr: torch.Tensor) -> torch.Tensor:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = torch.randint(low=0, high=upper_bound, size=(batch_size,))
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(
        self,
        batch_inds: torch.Tensor,
    ):
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()


class RolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_dim: int,
        action_dim: int,
        device: str = "cuda",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        num_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_dim, action_dim, device, num_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = torch.zeros(
                (self.buffer_size, self.num_envs, self.observation_dim), dtype=torch.float32
        )
        self.actions = torch.zeros(
            (self.buffer_size, self.num_envs, self.action_dim), dtype=torch.float32
        )
        self.rewards = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float32)
        self.returns = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float32)
        self.episode_starts = torch.zeros(
            (self.buffer_size, self.num_envs), dtype=torch.float32
        )
        self.values = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float32)
        self.advantages = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: torch.Tensor
    ) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().flatten()  # type: ignore[assignment]
        dones = dones.clone().cpu().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.to(torch.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add( # type: ignore
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)


        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.num_envs, self.action_dim))

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.clone().cpu().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = torch.randperm(self.buffer_size * self.num_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.num_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.num_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: torch.Tensor,
    ) -> RolloutBufferSamples:
        data = RolloutBufferSamples(
            self.observations[batch_inds].to(self.device),
            self.actions[batch_inds].to(self.device),
            self.values[batch_inds].flatten().to(self.device),
            self.log_probs[batch_inds].flatten().to(self.device),
            self.advantages[batch_inds].flatten().to(self.device),
            self.returns[batch_inds].flatten().to(self.device),
        )
        return data
