import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


# Initialise the environment
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8, seed=142), is_slippery = False, render_mode=None)

# Reset the environment to generate the first observation
# observation, info = env.reset(seed=46)
q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.9
gamma = 0.9

max_epislon = 1
min_epislon = 0.001
decay_rate = 0.0001

constant_epislon = 0.1


episodes_number = 15000
rewards_per_episode = np.zeros(episodes_number)

for episodes in range(episodes_number):
    terminated = False
    truncated = False
    state = env.reset()[0]
    while (not terminated and not truncated):
        rand = np.random.rand()
        actions = np.array([0, 1, 2, 3])
        # epislon = constant_epislon
        epislon = max(min_epislon + (max_epislon - min_epislon)*(1-decay_rate*episodes), min_epislon)
        # epislon = min_epislon + (max_epislon - min_epislon)*np.exp(-decay_rate*episodes)
        if rand < 1 - epislon:
            action = np.argmax(q[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, terminated, truncated, info = env.step(action)
        q[state, action] += alpha*(reward + gamma*np.max(q[new_state,:]) - q[state, action])
        state = new_state
    rewards_per_episode[episodes] = reward



sum_rewards = np.zeros(episodes_number)
for t in range(episodes_number):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig('frozen_lake4x4.png')
print('over')
print(q)


# env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = False, render_mode="human")
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8, seed=142), is_slippery = False, render_mode="human")

observation, info = env.reset()
while True:
    state = 0
    terminated = False
    truncated = False
    while (not terminated and not truncated):
        action = np.argmax(q[state])
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
    if terminated or truncated:
        observation, info = env.reset()


env.close()

