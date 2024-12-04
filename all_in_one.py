import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import contextlib
from itertools import product
import pandas as pd
import seaborn as sns
from gymnasium.envs.toy_text.frozen_lake import generate_random_map



sns.set_theme(style="darkgrid")

def MC(gamma_in, decay_rate_in, env):
    pi = np.random.randint(0, 4, size = env.observation_space.n)
    q = np.zeros([env.observation_space.n, env.action_space.n])
    action_num = np.zeros([env.observation_space.n, env.action_space.n])
    gamma = gamma_in
    max_epislon = 1
    min_epislon = 0.001
    decay_rate = decay_rate_in
    episodes_number = 40000
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
            epislon = min_epislon + (max_epislon - min_epislon)*np.exp(-decay_rate*episodes)
            # epislon = max(min_epislon + (max_epislon - min_epislon)*(1-decay_rate*episodes), min_epislon)

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

    sum_rewards = np.zeros(episodes_number)
    for t in range(episodes_number):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    return sum_rewards

def QLearning(alpha_in, gamma_in, decay_rate_in, env):

    q = np.zeros([env.observation_space.n, env.action_space.n])
    alpha = alpha_in
    gamma = gamma_in
    max_epislon = 1
    min_epislon = 0.001
    decay_rate = decay_rate_in
    episodes_number = 30000
    rewards_per_episode = np.zeros(episodes_number)

    for episodes in range(episodes_number):
        terminated = False
        truncated = False
        state = env.reset()[0]
        while (not terminated and not truncated):
            rand = np.random.rand()
            actions = np.array([0, 1, 2, 3])
            # epislon = max(min_epislon + (max_epislon - min_epislon)*(1-decay_rate*episodes), min_epislon)
            epislon = min_epislon + (max_epislon - min_epislon)*np.exp(-decay_rate*episodes)
            if rand < 1 - epislon:
                action = np.argmax(q[state,:])
            else:
                action = np.random.choice(actions)

            new_state, reward, terminated, truncated, info = env.step(action)
            q[state, action] += alpha*(reward + gamma*np.max(q[new_state,:]) - q[state, action])
            state = new_state
        rewards_per_episode[episodes] = reward

    sum_rewards = np.zeros(episodes_number)
    for t in range(episodes_number):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    return sum_rewards

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

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
    with local_seed(32):
        seeds = np.random.choice(100, 10, replace=False)

    methods = ["MC", "QLearning"]
    method = methods[0]

    param_grid_md = {
        "gamma" : [0.4, 0.7, 0.9],
        "decay_rate" : [0.0001, 0.0003]
    }
    param_grid_q = {
        "alpha" : [0.5, 0.9],
        "gamma" : [0.6, 0.9],
        "decay_rate" : [0.0001, 0.0005]
    }
    if method == methods[0]:
        param_grid = param_grid_md
    else:
        param_grid = param_grid_q

    param_keys = list(param_grid.keys())
    search_space = list(generate_search_space(param_grid).items())

    print(f"I'm conducting {method} method. Search space is ")
    print("="*20)
    print(search_space)
    print("="*20)


    search_space_chunks = list(chunk(search_space, 1))

    for ss_chunk in search_space_chunks:
        print(f"ss_chunk: {ss_chunk}"+"="*20)
        results = []
        data = []
        for seed in seeds:
            print(f"seed is {seed}")
            result = hypothetical_task_execution(param_keys, ss_chunk, seed, method)
            results.append(result)
        
        for seed_id, reward in enumerate(results, start=1):
            iterations = np.arange(1, len(reward) + 1)
            data.append(pd.DataFrame({
                "iteration": iterations,
                "reward": reward,
                "seed": seed_id
            }))
        df = pd.concat(data, ignore_index=True)
        interval = 10
        df_sampled = df[df["iteration"] % interval == 0] # down sampling...

        plt.clf()
        sns.lineplot(data=df_sampled, x="iteration", y="reward", errorbar="ci")
        plt.title("Average Reward with Confidence Interval (10 Seeds)")
        plt.xlabel("Episodes")
        plt.ylabel("Returns over past 100 episodes")
        if method == "MC":
            plt.savefig(f"{method}_gamma{ss_chunk[0][1][0]}_decay{ss_chunk[0][1][1]}_all.png")  
        elif method == "QLearning":
            plt.savefig(f"{method}_alpha{ss_chunk[0][1][0]}_gamma{ss_chunk[0][1][1]}_decay{ss_chunk[0][1][2]}.png")  
            

def hypothetical_task_execution(param_keys, ss_chunk, seed_in, method):
    # env = gym.make("FrozenLake-v1", map_name = "8x8" , is_slippery = False, render_mode=None)
    env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8, seed=int(seed_in)), is_slippery = True, render_mode=None)

    for hyperparam_id, combination in ss_chunk:
        hyperparameters = {}
        for idx, param_value in enumerate(combination):
            param_key = param_keys[idx]
            hyperparameters[param_key] = param_value
        print(f"running with method {method}, hyperparameter {hyperparameters}, seed {seed_in}")
        if method == "MC":
            result = MC(hyperparameters["gamma"], hyperparameters["decay_rate"], env)
            print("Simulation completed.")
        elif method == "QLearning":
            result = QLearning(hyperparameters["alpha"], hyperparameters["gamma"], hyperparameters["decay_rate"], env)
            print("Simulation completed.")
        else:
            print("wrong method input.")
    return result

if __name__ == "__main__":
    main()