import os
import hydra
from stable_baselines3 import PPO
from env import TactoEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from omegaconf import OmegaConf
import numpy as np
from sb3_contrib import RecurrentPPO

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard during training.
    """
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        Called at every environment step. Logs episode metrics to TensorBoard when an episode ends.
        """
        if self.locals.get("dones")[-1]:
            self.logger.record("episode/coverage", self.locals.get("infos")[0]["coverage"])
            self.logger.record("episode/reward", self.locals.get("infos")[0]["acc_reward"])
            self.logger.record("episode/length", self.locals.get("infos")[0]["horizon_counter"])
        return True

@hydra.main(config_path="conf", config_name="RL")
def train(cfg):
    """
    Main training function for PPO or RecurrentPPO using Hydra for configuration management.
    Loads environment and model, sets up logging and callbacks, and runs training.
    Args:
        cfg: Hydra configuration object loaded from conf/RL.yaml
    """
    env = TactoEnv(cfg)
    save_dir = os.path.join('Training', 'Logs', cfg.RL.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "RL.yaml"))

    # Set up checkpointing and custom TensorBoard callback
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.RL.save_freq,
        save_path=save_dir,
        name_prefix="model",
    )
    coverage_callback = TensorboardCallback()

    # Choose algorithm class based on config
    if cfg.RL.algorithm == "RecurrentPPO":
        AlgoClass = RecurrentPPO
    else:
        AlgoClass = PPO

    # Load from checkpoint if specified, else initialize new model
    if cfg.RL.pretrain_model_path:
        model = AlgoClass.load(cfg.RL.pretrain_model_path, env, verbose=1, tensorboard_log=save_dir)
        model.learning_rate = cfg.RL.learning_rate
        # Optionally update other params if supported by your SB3 version
    else:
        model = AlgoClass(
            cfg.RL.policy_network, env, verbose=1, tensorboard_log=save_dir, policy_kwargs=dict(normalize_images=False), learning_rate=cfg.RL.learning_rate,
            n_steps=cfg.RL.ppo_params.n_steps, batch_size=cfg.RL.ppo_params.batch_size, n_epochs=cfg.RL.ppo_params.n_epochs, gamma=cfg.RL.ppo_params.gamma,
            gae_lambda=cfg.RL.ppo_params.gae_lambda, ent_coef=cfg.RL.ppo_params.ent_coef)

    # Train for the specified number of steps
    # Set this to your desired training steps 
    model.learn(total_timesteps=500_000, callback=[checkpoint_callback, coverage_callback], reset_num_timesteps=False)
    # Save the final trained model
    model.save(os.path.join(save_dir, "final_model"))

if __name__ == "__main__":
    train()
