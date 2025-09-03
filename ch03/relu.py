# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

"""
ReLU函数（Rectified Linear Unit）
ReLU函数在输入大于0时，直接输出该值；在输入小于等于0时，输出0
"""
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()
