import contextlib
from itertools import product
import numpy as np

def chunks(lst, n):
    """Yield successive n-sized chunks from lst.以n的长度给lst切片"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@contextlib.contextmanager  #上下文管理器
def local_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def generate_search_space(parameter_grid,
                          random_search=False,
                          random_search_num_options=20,
                          random_search_seed=42):
    combinations = list(product(*parameter_grid.values()))
    search_space = {i: combinations[i] for i in range(len(combinations))}

    if random_search:
        if not random_search_num_options > len(search_space):
            reduced_space = {}
            with local_np_seed(random_search_seed):
                random_indices = np.random.choice(len(search_space), random_search_num_options, replace=False)
                for random_index in random_indices:
                    reduced_space[random_index] = search_space[random_index]
            search_space = reduced_space
    return search_space

def construct_search_spaces(param_grid):
    parameter_search_spaces = {}
    combinations = list(product(*param_grid.values()))
    search_space = {}
    for i in range(len(combinations)):
        k = str(i)
        v = dict(zip(list(param_grid.keys()), combinations[i]))
        search_space[k] = v

    return parameter_search_spaces

def main():
    # package 1 hyperparameter config and 2 seeds in same task.
    hyps_chunk_size = 1
    seeds_chunk_size = 2

    model_seeds = [0, 42, 84, 126, 168]
    parameter_grid = {
        "learning_rate": [0.01, 0.005, 0.001, 0.0005],
        "batch_size": [16],
        "num_layers": [2, 3, 4],
    }

    param_keys = list(parameter_grid.keys())
    local_tasks = []
    search_space = list(generate_search_space(parameter_grid).items())
    #search_space = list(generate_search_space(parameter_grid, random_search=True, random_search_num_options=5).items())
    print(f"my search space is")
    print("=" * 20)
    print(search_space)
    print("=" * 20)


    search_space_chunks = list(chunks(search_space, hyps_chunk_size))
    model_seed_chunks = list(chunks(model_seeds, seeds_chunk_size))

    task_id = 0
    for ss_chunk in search_space_chunks:
        for ms_chunk in model_seed_chunks:
            print(ss_chunk)
            print(ms_chunk)
            # save param_keys, ss_chunk, ms_chunk, and whatever other information in a json file.
            # which contains the task_id
            pass

            ## on the other side: read in param_keys, ss_chunk, ms_chunk from file
            hypothetical_task_execution(param_keys, ss_chunk, ms_chunk)


def hypothetical_task_execution(param_keys, ss_chunk, ms_chunk):
    for hyperparams_id, combination in ss_chunk:
        hyperparams = {}

        for idx, param_value in enumerate(tuple(combination)):
            param_key = param_keys[idx]
            hyperparams[param_key] = param_value

        for model_seed in ms_chunk:
            # inside this loop, create agent, run with model seed and hyperparams as usual.
            print(f"yahoo, running with seed {model_seed} and hyperparams {hyperparams}")
            pass


if __name__ == "__main__":
    main()