# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

"""
阶跃函数
当输入超过0时，输出1，否则输出0
"""
"""
阶跃函数
当输入超过0时，输出1，否则输出0
"""
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function1(x):
    y = x > 0
    """
    NumPy技巧：
    astype()方法通过参数指定期望的类型
    Python中将布尔型转换为int型后，True会转换为1，False会转换为0
    """
    return y.astype(int)

def step_function(x):
    return np.array(x > 0, dtype=int)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定y轴范围
plt.show()
