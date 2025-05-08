import os
import hydra
from stable_baselines3 import PPO
from env import TactoEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.logger import configure

@hydra.main(config_path="conf", config_name="test")
def test(cfg, num_repeat=1):
    """
    Evaluation/testing function for a trained PPO or RecurrentPPO model.
    Loads the environment and model, sets up logging, and runs evaluation episodes.
    Args:
        cfg: Hydra configuration object loaded from conf/test.yaml
        num_repeat: Number of evaluation episodes to run (default: 1)
    """
    env = TactoEnv(cfg)
    # Set up TensorBoard logger in outputs/logs
    log_dir = cfg.RL.save_dir if hasattr(cfg.RL, 'save_dir') else "outputs/logs"
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    input("press any  key to start")
    # Load the appropriate model type based on config
    if cfg.RL.algorithm == "RecurrentPPO":
        model = RecurrentPPO.load(cfg.RL.pretrain_model_path, env)
    else:
        model = PPO.load(cfg.RL.pretrain_model_path, env)
    model.set_logger(new_logger)

    for i in range(num_repeat):
        print("##### Iteration: ", i)
        # Run evaluation for 5000 steps (or until episode ends)
        model.learn(total_timesteps=5000, reset_num_timesteps=True)

if __name__ == "__main__":
    test()