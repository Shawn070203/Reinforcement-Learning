import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Initialise the environment
env = gym.make("FrozenLake-v1", map_name = "4x4" , is_slippery = False, render_mode=None)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.8
gamma = 0.5

max_epislon = 1
min_epislon = 0.001
decay_rate = 0.001

constant_epislon = 0.2


episodes_number = 2000
rewards_per_episode = np.zeros(episodes_number)

for episodes in range(episodes_number):
    t = -1
    terminated = False
    truncated = False
    state = env.reset()[0]
    while (not terminated and not truncated):
        t += 1
        rand = np.random.rand()
        actions = np.array([0, 1, 2, 3])
        # epislon = constant_epislon
        epislon = min_epislon + (max_epislon - min_epislon)*np.exp(-decay_rate*episodes)
        if rand < 1 - epislon:
            action = np.argmax(q[state])
        else:
            action = np.random.choice(actions)

        new_state, reward, terminated, truncated, info = env.step(action)
        q[state, action] += alpha*(reward + gamma*np.max(q[new_state]) - q[state, action])
        state = new_state
    rewards_per_episode[episodes] = reward



sum_rewards = np.zeros(episodes_number)
for t in range(episodes_number):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig('frozen_lake4x4.png')
print('over')


# env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = False, render_mode="human")
# observation, info = env.reset(seed=35)
# while True:
#     state = 0
#     terminated = False
#     truncated = False
#     while (not terminated and not truncated):
#         action = pi[state]
#         new_state, reward, terminated, truncated, info = env.step(action)
#         state = new_state
#     if terminated or truncated:
#         observation, info = env.reset()


env.close()

