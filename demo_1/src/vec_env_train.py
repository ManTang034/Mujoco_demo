import argparse
import logging
import os

from envs.ball_balance_env import BallBalanceEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def initialize_environment(model_path, render_mode):
    """Initialize the environment and check its compatibility."""
    logging.info("Initializing environment...")
    try:
        vec_env = make_vec_env(
            lambda: BallBalanceEnv(render_mode=render_mode, model_path=model_path),
            n_envs=4,  # 并行环境数量
            seed=42,  # 随机种子
        )
        # check_env(vec_env)
        return vec_env
    except Exception as e:
        logging.error(f"Failed to initialize environment: {e}")
        exit(1)


def create_model(env, log_path, device="cpu"):
    """Create and return a PPO model."""
    logging.info("Creating PPO model...")
    try:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.join(log_path, "sac_ball_balance_tensorboard/"),
            device=device,
        )
        return model
    except Exception as e:
        logging.error(f"Failed to create model: {e}")
        exit(1)


def train_model(model, env, total_timesteps, model_path, save_path):
    """Train the model and save the best version."""
    logging.info("Starting training...")
    try:
        eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(model_path, "sac_best/"),
            eval_freq=10000,
            deterministic=True,
            render=False,
            warn=True,
        )
        model.learn(
            total_timesteps=total_timesteps, callback=eval_callback, log_interval=4
        )
        model.save(os.path.join(save_path, "sac_ball_balance"))
        logging.info("Training completed and model saved.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        exit(1)


def main(args):
    """Main function to initialize and train the model."""
    # Initialize the environment
    env = initialize_environment(args.model_path, args.render_mode)

    # Create the model
    model = create_model(env, args.log_path, device="cpu")

    # Train the model
    train_model(model, env, args.total_timesteps, args.save_path, args.save_path)


if __name__ == "__main__":
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Train a PPO model for BallBalanceEnv."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(project_root, "assets/models/ball_balance.xml"),
        help="Path to the MuJoCo model file.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=os.path.join(project_root, "data/logs/"),
        help="Path to save training logs and TensorBoard data.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(project_root, "data/models/"),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=150000,
        help="Total number of timesteps for training.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        choices=["human", "rgb_array"],
        help="Rendering mode: 'human' for interactive window, 'rgb_array' for capturing frames.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ensure the output directories exist
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    # Call the main function
    main(args)
