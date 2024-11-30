import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import contextlib
from itertools import product

def MC(gamma_in, decay_rate_in, seed, env):
    pi = np.random.randint(0, 4, size = env.observation_space.n)
    q = np.zeros([env.observation_space.n, env.action_space.n])
    action_num = np.zeros([env.observation_space.n, env.action_space.n])
    gamma = gamma_in
    max_epislon = 1
    min_epislon = 0.001
    decay_rate = decay_rate_in
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
            rand = np.random.rand()
            actions = np.array([0, 1, 2, 3])
            # epislon = min_epislon + (max_epislon - min_epislon)*np.exp(-decay_rate*episodes)
            epislon = max(min_epislon + (max_epislon - min_epislon)*(1-decay_rate*episodes), min_epislon)

            if rand < 1 - epislon:
                action = pi[state]
            else:
                action = np.random.choice(actions)
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

def QLearning(alpha_in, gamma_in, decay_rate_in, seed, env):
    q = np.zeros([env.observation_space.n, env.action_space.n])
    alpha = alpha_in
    gamma = gamma_in
    max_epislon = 1
    min_epislon = 0.001
    decay_rate = decay_rate_in
    episodes_number = 15000
    rewards_per_episode = np.zeros(episodes_number)

    for episodes in range(episodes_number):
        terminated = False
        truncated = False
        state = env.reset()[0]
        while (not terminated and not truncated):
            rand = np.random.rand()
            actions = np.array([0, 1, 2, 3])
            epislon = max(min_epislon + (max_epislon - min_epislon)*(1-decay_rate*episodes), min_epislon)
            # epislon = min_epislon + (max_epislon - min_epislon)*np.exp(-decay_rate*episodes)
            if rand < 1 - epislon:
                action = np.argmax(q[state,:])
            else:
                action = np.random.choice(actions)

            new_state, reward, terminated, truncated, info = env.step(action)
            q[state, action] += alpha*(reward + gamma*np.max(q[new_state,:]) - q[state, action])
            state = new_state
        rewards_per_episode[episodes] = reward

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i, i+n]

@contextlib.contextmanager
def local_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def generate_search_space(parameter_grid,
                          random_search=False,
                          random_num_search=20,
                          random_search_seed=42):
    combinations = list(product(*parameter_grid.values()))
    search_space = {i: combinations[i] for i in range(len(combinations))}

    if random_search:
        if random_num_search <= len(combinations):
            space_reduced = {}
            with local_seed(random_search_seed):
                random_indices = np.random.choice(len(search_space), random_num_search, replace=False)
                for index in random_indices:
                    space_reduced[index] = search_space[index]
            search_space = space_reduced
    return search_space

def main():
    with local_seed(42):
        seeds = np.random.choice(100, 10, replace=False)

    methods = ["MC", "QLearning"]
    method = methods[0]

    param_grid_md = {
        "gamma" : [0.2, 0.8, 1],
        "decay_rate" : [0.001, 0.0005]
    }
    param_grid_q = {
        "alpha" : [0.2, 0.5, 0.8],
        "gamma" : [0.2, 0.8, 1],
        "decay_rate" : [0.001, 0.0005]
    }

    param_keys = list(param_grid_md.keys())
    search_space = list(generate_search_space(param_grid_md).items)

    print(f"I'm conducting {method} method. Search space is ")
    print("="*20)
    print(search_space)
    print("="*20)


    search_space_chunks = list(chunk(search_space, 1))

    for ss_chunk in search_space_chunks:
        for seed in seeds:
            print(f"ss_chunk: {ss_chunk}")
            print(f"seed is {seed}")
            hypothetical_task_execution(param_keys, ss_chunk, seed, method)


def hypothetical_task_execution(param_keys, ss_chunk, seed, method):
    env = gym.make("FrozenLake-v1", map_name = "4x4" , is_slippery = False, render_mode=None)
    observation, info = env.reset(seed=35)

    for hyperparam_id, combination in ss_chunk:
        hyperparameters = {}
        for idx, param_value in enumerate(combination):
            param_key = param_keys[idx]
            hyperparameters[param_key] = param_value
        print(f"running with method {method}, hyperparameter {hyperparameters}, seed {seed}")
        if method == "MC":
            MC(hyperparameters["gamma"], hyperparameters["decay_rate"], seed, env)
        elif method == "QLearning":
            QLearning(hyperparameters["alpha"], hyperparameters["gamma"], hyperparameters["decay_rate"], seed, env)
        else:
            print("wrong method input.")

if __name__ == "__main__":
    main()