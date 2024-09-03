import hydra
from omegaconf import DictConfig
from models.rl.mdt_residual import MDTResidual
from models.rl.ppo import PPO
from models.rl.env import make_playtable_env_func
from gym.vector import SyncVectorEnv




@hydra.main(config_path="../conf", config_name="train_residual")
def train(conf: DictConfig):
    policy: MDTResidual = hydra.utils.instantiate(conf.policy)
    policy.configure_mdt()
    make_env_fns = iter([make_playtable_env_func(conf.env)])
    env = SyncVectorEnv(make_env_fns)
    trainer = PPO(
        env = env,


    )
            

