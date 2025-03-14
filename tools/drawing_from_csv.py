import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data1 = pd.read_csv('./DQN_results/raw_data/8x8FReward.csv')
# data2 = pd.read_csv('8x8TR2.csv')
# data3 = pd.read_csv('8x8TR3.csv')
# data4 = pd.read_csv('8x8TR4.csv')

# 绘制损失曲线
plt.plot(data1['Step'], data1['Value'], label='8x8_Not_slippery')
# plt.plot(data2['Step'], data2['Value'], label='8x8_slippery_2')
# plt.plot(data3['Step'], data3['Value'], label='8x8_slippery_3')
# plt.plot(data4['Step'], data4['Value'], label='8x8_slippery_4')
plt.xlabel('TimeStep')
plt.ylabel('Value')
plt.title('Training Process')
plt.legend()
plt.show()


