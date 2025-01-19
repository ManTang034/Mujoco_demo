import gymnasium as gym
from envs.pusher_v5 import PusherEnv
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

env = PusherEnv(
    render_mode="rgb_array",
    xml_file="/home/ywang034/projects/mujoco_demo/demo_2/assets/models/pusher_v5.xml",
)
env = Monitor(env)
model = SAC("MlpPolicy", env, verbose=1, device="cpu")
model.learn(
    total_timesteps=60_000,
)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
