# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function0(x):
    """
    阶跃函数
    当输入超过0时，输出1，否则输出0

    缺点：
    参数x只能接受实数（浮点数）。也就是说，允许形如step_function(3.0)的调用，
    但不允许参数取NumPy数组，例如 step_function(np.array([1.0, 2.0]))
    """
    if x > 0:
        return 1
    else:
        return 0


def step_function1(x):
    y = x > 0

    """
    对NumPy数组进行不等号运算后，数组的各个元素都会进行不等号运算，生成一个布尔型数组。
    
    说明：
    数组x中大于0的元素被转换为True
    小于等于0的元素被转换为False
    """
    print(y)

    """
    NumPy技巧：
    astype()方法通过参数指定期望的类型
    Python中将【布尔类型】转换为int型后，True会转换为1，False会转换为0
    """
    return y.astype(int)


def step_function(x):
    return np.array(x > 0, dtype=int)


def _main():
    X = np.arange(-5.0, 5.0, 0.1)  # 生成-5到5的0.1的间隔数据

    # Y = step_function0(X)
    # Y = step_function1(X)
    Y = step_function(X)

    plt.plot(X, Y)
    plt.ylim(-0.1, 1.1)  # 指定y轴范围
    plt.show()


if __name__ == '__main__':
    _main()
