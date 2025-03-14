import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 10 seeds data. 15000 simulations(episodes) for each data
np.random.seed(42)
data = {
    "iteration": np.tile(np.arange(1, 15001), 10),  
    "reward": np.concatenate([np.cumsum(np.random.randn(15000) + 0.1) for _ in range(10)]),  
    "seed": np.repeat(np.arange(1, 11), 15000)  # 种子编号
}
df = pd.DataFrame(data)

# plot using seaborn
sns.lineplot(data=df, x="iteration", y="reward", ci="sd")  
plt.title("Average Reward over 10 Seeds with Confidence Interval")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.show()
