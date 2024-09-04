from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Iterable

import cloudpickle
import gym
import numpy as np
import torch

from mdt.models.rl.env import PlayTableTaskEnv


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import

    parent_remote.close()
    env = env_fn_wrapper.var()
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                # convert to SB3 VecEnv api
                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                observation, reset_info = env.reset()
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(PlayTableTaskEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(
        self,
        make_env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = "forkserver",
    ):
        self.waiting = False
        self.closed = False
        self.num_envs = len(make_env_fns)

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, make_env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.is_vector_env = True

    def _add_info(self, infos, env_info, index):
        pass

    def reset(self):  # type: ignore
        """Resets each of the sub-environments and concatenate the results together.

        Returns:
            Concatenated observations and info from each sub-environment
        """
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, info = zip(*results)  # type: ignore[assignment]
        # Seeds and options are only used once
        return self._flatten_obs(obs), info

    def step_async(self, actions: torch.Tensor):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return (
            self._flatten_obs(obs),
            torch.Tensor(rewards),
            torch.Tensor(dones),
            infos,
        )

    def step(self, actions: torch.Tensor):  # type: ignore
        self.step_async(actions)
        return self.step_wait()

    def _flatten_obs(
        self, obs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        result = {}
        for d in obs:
            for key, tensor in d.items():
                if key not in result:
                    result[key] = []
                result[key].append(tensor)
        for key in result:
                result[key] = torch.cat(result[key], dim=0)
        return result

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()
    def _get_target_remotes(self, indices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]
    def _get_indices(self, indices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

# class DummyVecEnv(PlayTableTaskEnv):
#     def __init__(
#         self,
#         make_env_fns: List[Callable[[], gym.Env]],
#         ):
#         self.env = [make_env_fn() for make_env_fn in make_env_fns]
#         self.num_envs = len(self.env)
#         self.is_vector_env=True
#         
#     def reset():




class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)
