import contextlib
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List

import einops
import gym
import hydra
import numpy as np
import pyhash
import torch
from omegaconf import DictConfig

from calvin_env.calvin_env.envs.play_table_env import PlayTableSimEnv
from calvin_env.calvin_env.envs.tasks import Tasks
from mdt.models.rl.utils import TASK_TO_LABEL

hasher = pyhash.fnv1_32()


class PlayTableTaskEnv(PlayTableSimEnv):
    def __init__(
        self,
        cameras: DictConfig,
        robot_cfg: DictConfig,
        scene_cfg: DictConfig,
        task_cfg: DictConfig,
        task_annotation: DictConfig,
        observation_space_keys: DictConfig,
        seed: int,
        use_vr: bool,
        bullet_time_step: float,
        show_gui: bool,
        use_scene_info: bool,
        use_egl: bool,
        language_embedding_path: str,
        sequences: List,
        control_freq: int = 30,
        ep_len: int = 360,
    ):
        super().__init__(
            robot_cfg=robot_cfg,
            seed=seed,
            use_vr=use_vr,
            bullet_time_step=bullet_time_step,
            cameras=cameras,
            show_gui=show_gui,
            scene_cfg=scene_cfg,
            use_scene_info=use_scene_info,
            use_egl=use_egl,
            control_freq=control_freq,
        )
        self.ep_len = ep_len
        self.num_steps = 0
        self.observation_space_keys = observation_space_keys
        self.task_annotation = task_annotation
        self.sequences = sequences
        self.task_oracle: Tasks = hydra.utils.instantiate(task_cfg)
        self.task2label = TASK_TO_LABEL
        self.label2task = {value: key for key, value in TASK_TO_LABEL.items()}
        self.language_embedding: Dict = np.load(
            Path(language_embedding_path), allow_pickle=True
        ).item()

        self.rewards = []
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_times = []
        self.t_start = time.time()

    def reset(self):  # type: ignore
        initial_state, eval_sequence = random.choice(self.sequences)
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        self.current_sequence = deque(eval_sequence)
        self.current_subtask = self.current_sequence.popleft()
        obs = super().reset(robot_obs=robot_obs, scene_obs=scene_obs)
        del obs["robot_obs"]
        del obs["scene_obs"]
        obs = self.process_obs(obs)
        self.start_info = super().get_info()
        self.rewards = []
        self.num_steps = 0
        return obs, self.start_info

    def step(self, action: torch.Tensor):  # type: ignore
        action = action.clone().cpu().numpy()
        action = action.squeeze()
        action[-1] = 1 if action[-1] > 0 else -1
        obs, reward, _, info = super().step(action)
        self.num_steps += 1
        del obs["robot_obs"]
        del obs["scene_obs"]
        current_task_info = self.task_oracle.get_task_info_for_set(
            self.start_info, info, {self.current_subtask}
        )
        # if comepleted current task, get reward, and do a new task
        if len(current_task_info) > 0:
            reward += 1
            if self.current_sequence:
                self.current_subtask = self.current_sequence.popleft()
                self.start_info = super().get_info()
        self.rewards.append(reward)
        obs = self.process_obs(obs)
        # done
        if not self.current_sequence and sum(self.rewards) == 5:
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
            }
            new_obs, new_info = self.reset()
            new_info["terminal_observation"] = obs
            new_info["TimeLimit.truncated"] = False
            new_info["ep_info"] = ep_info
            return new_obs, reward, True, new_info
        # truncate
        if self.num_steps > self.ep_len:
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
            }
            new_obs, new_info = self.reset()
            new_info["terminal_observation"] = obs
            new_info["TimeLimit.truncated"] = True
            new_info["ep_info"] = ep_info
            return new_obs, reward, True, new_info

        return obs, reward, False, info

    def process_obs(self, obs) -> Dict[str, torch.Tensor]:
        """Flatten a nested dict numpy array into a dict of  tensors
            Also rearrange image channels (w h c) -> (c w h)

        Args:
            obs : a nested dict of numpy arrays

        Returns:
            a dict of tensors
        """
        return_obs = {}
        for key, child_obs in obs.items():
            if isinstance(child_obs, dict):
                child_items = self.process_obs(child_obs)
                return_obs.update(child_items)
            else:
                result = torch.Tensor(child_obs)
                if len(result.shape) == 3:
                    result = einops.rearrange(result, "w h c -> c w h")
                return_obs[key] = result.unsqueeze(0)

        goal_text = self.task_annotation[self.current_subtask][0]
        return_obs ["goal_label"] = torch.Tensor([self.task2label[goal_text]])
        return_obs ["goal_emb"] = torch.from_numpy(
            self.language_embedding[self.current_subtask]["emb"]
        ).squeeze(1)
        return return_obs


def make_playtable_env_func(conf: DictConfig, sequences: List) -> Callable[[], gym.Env]:
    def make_env_fn():
        env = PlayTableTaskEnv(
            robot_cfg=conf.robot,
            seed=conf.env.seed,
            use_vr=conf.env.use_vr,
            bullet_time_step=conf.env.bullet_time_step,
            cameras=conf.cameras,
            show_gui=conf.env.show_gui,
            scene_cfg=conf.scene,
            use_scene_info=conf.env.use_scene_info,
            use_egl=conf.env.use_egl,
            task_annotation=conf.env.task_annotation,
            language_embedding_path=conf.env.language_embedding_path,
            task_cfg=conf.env.task_cfg,
            sequences=sequences,
            control_freq=conf.env.control_freq,
            observation_space_keys=conf.env.observation_space_keys,
        )
        return env

    return make_env_fn


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    # we want to have a "deterministic" random seed for each initial condition
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        pos_noise = np.random.uniform(low=[-1e-2, -1e-2, 0], high=[1e-2, 1e-2, 0])
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[6:9] += pos_noise
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[12:15] += pos_noise
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[18:21] += pos_noise
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
