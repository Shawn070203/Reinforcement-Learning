import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 选参
# Initialise the environment
env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = False, render_mode=None)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=35)

pi = np.random.randint(0, 4, size = env.observation_space.n)
q = np.random.rand(env.observation_space.n, env.action_space.n)
action_num = np.zeros([env.observation_space.n, env.action_space.n])
gamma = 1
max_epislon = 1
min_epislon = 0.001
decay_rate = 0.001
constant_epislon = 0.1
episodes_number = 15000
rewards_per_episode = np.zeros(episodes_number)

for episodes in range(episodes_number):
    t = -1
    experience = []
    rewards = [0]
    terminated = False
    truncated = False
    state = env.reset()[0]
    while (not terminated and not truncated):

        t += 1
    # this is where you would insert your policy
        rand = np.random.rand()
        actions = np.array([0, 1, 2, 3])
        epislon = constant_epislon
        # epislon = min_epislon + (max_epislon - min_epislon)*np.exp(-decay_rate*episodes)
        epislon = max(min_epislon + (max_epislon - min_epislon)*(1-decay_rate*episodes), min_epislon)

        if rand < 1 - epislon:
            action = pi[state]
        else:
            action = np.random.choice(actions)
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
        new_state, reward, terminated, truncated, info = env.step(action)
        experience.append((state, action))
        action_num[state, action] += 1
        rewards.append(reward)
        state = new_state
    rewards_per_episode[episodes] = rewards[-1]
    G = 0
    experience_np = np.array(experience)
    rewards_np = np.array(rewards)
    for _ in range(t, -1, -1):
        G = gamma*G + rewards_np[_+1]
        exist = any(np.array_equal(np.array(experience_np[_]), a) for a in np.array(experience_np[0: _]))
        if not exist:
            state_episode = experience_np[_][0]
            action_episode = experience_np[_][1]
            q[state_episode, action_episode] += 1/action_num[state_episode, action_episode]*(G - q[state_episode, action_episode])
            pi[state_episode] = np.argmax(np.array(q[state_episode]))

    # If the episode has ended then we can reset to start a new episode





# 结果体现
sum_rewards = np.zeros(episodes_number)
for t in range(episodes_number):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig('frozen_lake8x8.png')
print('over')
print(pi)

env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = False, render_mode="human")
observation, info = env.reset(seed=35)
while True:
    state = 0
    terminated = False
    truncated = False
    while (not terminated and not truncated):
        action = pi[state]
        new_state, reward, terminated, truncated, info = env.step(action)
        state = new_state
    if terminated or truncated:
        observation, info = env.reset()


env.close()

