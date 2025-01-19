import argparse
import logging
import os
from threading import Thread

import cv2
import imageio
from envs.pusher_v5 import PusherEnv
from stable_baselines3 import A2C, PPO, SAC

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_gif_async(frames, output_dir):
    """Save frames as a GIF in a separate thread."""
    gif_path = os.path.join(output_dir, "test_pusher_v5.gif")
    with imageio.get_writer(gif_path, mode="I") as writer:
        for frame in frames:
            writer.append_data(frame)
    logging.info(f"GIF saved to {gif_path}")


def run_model(env, model, num_steps, frame_interval, render_mode, save_gif, output_dir):
    """Run the model and render the environment."""
    obs, info = env.reset()
    frames = []

    for step in range(num_steps):
        action, state_info = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        logging.debug(f"Step {step}: Action = {action}, Reward = {reward}")

        if render_mode == "rgb_array":
            image = env.render()
            if step % frame_interval == 0:
                frames.append(image)
            cv2.imshow("Pusher_v5 Environment", image)
            cv2.waitKey(1)

        if done or truncated:
            obs, info = env.reset()

    if save_gif and render_mode == "rgb_array":
        gif_thread = Thread(target=save_gif_async, args=(frames, output_dir))
        gif_thread.start()

    env.close()
    if render_mode == "rgb_array":
        cv2.destroyAllWindows()


def main(args):
    """Main function to initialize and run the model."""
    logging.info("Initializing environment and model...")

    try:
        # Initialize the environment
        env = PusherEnv(render_mode=args.render_mode, xml_file=args.assets_path)

        # Load the model
        model = A2C.load(args.model_path, device="cpu")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Error initializing environment or loading model: {e}")
        exit(1)

    # Run the model
    run_model(
        env,
        model,
        args.num_steps,
        args.frame_interval,
        args.render_mode,
        args.save_gif,
        args.output_dir,
    )


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run a trained PPO model on BallBalanceEnv."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(project_root, "data/models/a2c_push_v5.zip"),
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--assets_path",
        type=str,
        default=os.path.join(project_root, "assets/models/pusher_v5.xml"),
        help="Path to the MuJoCo model file.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=500,
        help="Number of steps to run the environment.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="Interval for capturing frames (e.g., capture every 5th frame).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "media"),
        help="Directory to save the output GIF.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        choices=["human", "rgb_array"],
        help="Rendering mode: 'human' for interactive window, 'rgb_array' for capturing frames.",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Whether to save the rendered frames as a GIF (only applicable if render_mode is 'rgb_array').",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Call the main function
    main(args)
