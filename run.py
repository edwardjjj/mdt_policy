from multiprocessing import freeze_support
from pathlib import Path
from mdt.rollout.rollout_video import RolloutVideo

import click
import hydra
import numpy as np
import torch
import wandb

from mdt.evaluation.multistep_sequences import get_sequences
from mdt.models.rl.env import PlayTableTaskEnv, make_playtable_env_func
from mdt.models.rl.mdtv_residual import MDTVResidual
from mdt.models.rl.ppo import PPO
from mdt.models.rl.vector_env import SubprocVecEnv


def eval_policy(policy: MDTVResidual, env: PlayTableTaskEnv, num_episodes: int) -> None:
    policy.reset()
    ep_rewards = np.zeros(6)
    for i in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                actions, _, _, _, _ = policy(obs)
            obs, reward, done, info = env.step(actions)
            episode_reward += reward
        ep_rewards[episode_reward] += 1
    result = {}
    for i in range(1, 6):
        result[f"completed {i} tasks"] = np.sum(ep_rewards[i:6]) / np.sum(
            ep_rewards[0:6]
        )
    wandb.log(result)


def eval(eval_config, env_config, eval_ckpt_path) -> None:
    run = wandb.init(project="residual_calvin_eval", entity="aklab", tags=["mdtv", "d"])
    sequences = get_sequences(env_config.env.num_sequences)
    env = make_playtable_env_func(env_config, sequences)()
    policy: MDTVResidual = hydra.utils.instantiate(
        eval_config.policy, _recursive_=False
    )
    policy.configure_mdtv()
    policy.load_state_dict(torch.load(eval_ckpt_path, weights_only=True))
    eval_policy(policy, env, eval_config.num_episodes)


def train(config, env_config, eval_episodes) -> None:
    run = wandb.init(
        project="residual_calvin_train", entity="aklab", tags=["mdtv", "d"]
    )
    sequences = get_sequences(env_config.env.num_sequences)
    train_env = SubprocVecEnv(
        [make_playtable_env_func(env_config, sequences) for _ in range(config.num_envs)]
    )
    policy: MDTVResidual = hydra.utils.instantiate(config.policy, _recursive_=False)
    policy.configure_mdtv()
    policy.to(torch.device("cuda"))
    ppo: PPO = hydra.utils.instantiate(config.ppo, policy=policy, env=train_env)
    ppo.learn(config.num_training_steps, log_interval=config.log_interval)
    train_env.close()
    eval_env:PlayTableTaskEnv = make_playtable_env_func(env_config, sequences)() # type:ignore
    eval_policy(policy, eval_env, eval_episodes)
    torch.save(
        policy.state_dict(),
        f"/home/edward/projects/mdt_policy/checkpoints/{run.id}.ckpt",
    )


@click.command()
@click.option("--eval", default=False, help="evaluate policy")
@click.option(
    "--eval_ckpt_path", default=None, help="path to checkpoint to be evaluated"
)
@click.option(
    "--config_name", default="train_residual_calvin_d", help="config name for training"
)
@click.option(
    "--eval_episodes", default=20, help="number of eval episodes after training"
)
def main(eval, eval_ckpt_path, config_name, eval_episodes):
    if eval:
        if eval_ckpt_path is None:
            raise ValueError("eval_ckpt_path can't be None")
        else:
            eval_ckpt_path = Path(eval_ckpt_path)
            assert eval_ckpt_path.exists(), f"{eval_ckpt_path} does not exist"
    hydra.initialize(config_path="conf")
    config = hydra.compose(config_name=config_name)
    env_config = hydra.compose(config_name="calvin_env_d")
    if eval:
        eval(config, env_config, eval_ckpt_path)
    else:
        train(config, env_config, eval_episodes)


if __name__ == "__main__":
    freeze_support()
    main()
