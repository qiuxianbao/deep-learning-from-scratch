# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# データの作成
x = np.arange(0, 6, 0.1) # 0から6まで0.1刻みで生成， 以0.1为单位，生成0到6的数据
y = np.sin(x)

# グラフの描画
plt.plot(x, y) # 绘制图形
plt.show()