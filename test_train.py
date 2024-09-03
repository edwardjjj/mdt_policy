import hydra
import torch
from multiprocessing import freeze_support
from mdt.models.rl.ppo import PPO
from mdt.models.rl.env import make_playtable_env_func
from mdt.models.rl.vector_env import SubprocVecEnv
from mdt.models.rl.mdtv_residual import MDTVResidual
from omegaconf import DictConfig
from mdt.evaluation.multistep_sequences import get_sequences
import wandb


@hydra.main(version_base="1.1", config_path="./conf", config_name="train_residual_calvin_abcd")
def main(config: DictConfig):
    wandb.init(project="residual_calvin", entity="aklab", tags=["mdtv", "abcd"])

    env_config = hydra.compose(config_name="calvin_env_abcd")
    sequences = get_sequences(env_config.env.num_sequences, 100)
    env = SubprocVecEnv([make_playtable_env_func(env_config, sequences) for _ in range(2)])
    policy: MDTVResidual = hydra.utils.instantiate(config.policy, _recursive_=False)
    policy.configure_mdtv()
    policy.to(torch.device("cuda"))
    ppo = PPO(
        env=env,
        observation_dim=903,
        action_dim=7,
        buffer_size=256,
        policy=policy,
        device="cuda",
        n_steps=256,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    ppo.learn(100_000, log_interval=10)

if __name__ == "__main__":
    freeze_support()
    main()

