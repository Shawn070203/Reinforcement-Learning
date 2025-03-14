import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib as plt


model_dir = "models"
log_dir = "logs_1"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env):
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device='cuda')
    TIMESTEPS = 1500000
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{model_dir}/DQN_{TIMESTEPS}")

    

def test(env, path_to_model):
    model = DQN.load(path_to_model, env=env)
    obs = env.reset()[0]
    test_steps = 500
    while True:
        action, next_state = model.predict(obs, deterministic=True)
        obs, terminated, truncated, _, _ = env.step(action.item())

        if terminated or truncated:
            test_steps -= 1
            
            if test_steps <= 0:
                break



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--train', action='store_true')
    # parser.add_argument('-s', '--test', metavar='path_to_models')
    # args = parser.parse_args()

    # if args.train:
    #     env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = True, render_mode=None)
    #     train(env)

    # if args.test:
    #     if os.path.isfile(args.test):
    #         env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = False, render_mode='human')
    #         test(env, path_to_model=args.test)
    #     else:
    #         print(f'{args.test} not found')
    env = Monitor(gym.make("FrozenLake-v1", desc=generate_random_map(size=8, seed=79), is_slippery = False, render_mode=None))
    train(env)



