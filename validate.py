import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Initialise the environment
env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = False, render_mode="none")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

pi = np.random.randint(0, 4, size = env.observation_space.n)
q = [[0.5, 0.5, 0.5, 0.5] for _ in range(env.observation_space.n)]
returns = [[[], [], [], []] for _ in range(env.observation_space.n)]
gamma = 0.9
epislon = 0.5
episodes_number = 5000
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
        if rand < 1 - epislon + epislon / 4:
            action = pi[state]
        else:
            actions = actions[~np.isin(actions, np.array(pi[state]))]
            action = np.random.choice(actions)
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
        new_state, reward, terminated, truncated, info = env.step(action)
        experience.append((state, action))
        rewards.append(reward)
        state = new_state
    rewards_per_episode[episodes] = rewards[-1]
    G = 0
    experience_np = np.array(experience)
    rewards_np = np.array(rewards)
    for _ in range(t, -1, -1):
        G = gamma*G + rewards[_+1]
        a = np.array(experience_np[0: _])
        b = experience_np[_]
        # exist = any(np.array_equal(np.array(experience_np[_]), a) for a in np.array(experience_np[0: _]))
        # if not exist:
        state_episode = experience_np[_][0]
        action_episode = experience_np[_][1]
        returns[state_episode][action_episode].append(G)
        q[state_episode][action_episode] = np.average(np.array(returns[state_episode][action_episode]))
        pi[state_episode] = np.argmax(np.array(q[state_episode]))

    # If the episode has ended then we can reset to start a new episode


env.close()
sum_rewards = np.zeros(episodes_number)
for t in range(episodes_number):
    sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
plt.plot(sum_rewards)
plt.savefig('frozen_lake8x8.png')
print('over')
print(pi)