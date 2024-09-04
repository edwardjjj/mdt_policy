from multiprocessing import freeze_support
from pathlib import Path

import click
import hydra
import numpy as np
import torch
import wandb

from mdt.evaluation.multistep_sequences import get_sequences
from mdt.models.rl.env import make_playtable_env_func
from mdt.models.rl.mdtv_residual import MDTVResidual
from mdt.models.rl.ppo import PPO
from mdt.models.rl.vector_env import SubprocVecEnv


def eval_policy(policy: MDTVResidual, env, num_episodes) -> None:
    policy.reset()
    obs, info = env.reset()
    ep_rewards = np.zeros(6)
    for i in range(num_episodes):
        current_reward = 0
        while True:
            with torch.no_grad():
                actions, _, _, _, _ = policy(obs)
            obs, reward, done, info = env.step(actions)
            current_reward += reward
            if done:
                ep_rewards[current_reward] += 1
                break
    result = {}
    for i in range(1, 6):
        result[f"completed {i} tasks"] = ep_rewards[i:6].sum() / ep_rewards[0:6].sum()
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


def train(config, env_config) -> None:
    run = wandb.init(
        project="residual_calvin_train", entity="aklab", tags=["mdtv", "d"]
    )
    sequences = get_sequences(env_config.env.num_sequences)
    train_env = SubprocVecEnv(
        [make_playtable_env_func(env_config, sequences) for _ in range(10)]
    )
    policy: MDTVResidual = hydra.utils.instantiate(config.policy, _recursive_=False)
    policy.configure_mdtv()
    policy.to(torch.device("cuda"))
    ppo = PPO(
        env=train_env,
        observation_dim=903,
        action_dim=7,
        buffer_size=10,
        policy=policy,
        device="cuda",
        n_steps=10,
        batch_size=10,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    ppo.learn(1_000, log_interval=10)
    train_env.close()
    eval_env = make_playtable_env_func(env_config, sequences)()
    eval_policy(policy, eval_env, 500)
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
def main(eval, eval_ckpt_path, config_name):
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
        train(config, env_config)


if __name__ == "__main__":
    freeze_support()
    main()
