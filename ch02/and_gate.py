# coding: utf-8
import numpy as np


def AND0(x1, x2):
    """
    感知机简单实现-与门
    步骤解析
    """
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def AND(x1, x2):
    """
    感知机，导入权重和偏置
    b称为偏置，w1和w2称为权重。有时也会将b、w1、w2这些参数统称为权重

    注意：
    w1和w2是控制输入信号的重要性的参数
    偏置b是调整神经元被激活的容易程度（输出信号为1的程度）的参数
    """
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7

    # tmp = np.sum(w*x) + b
    tmp = x.dot(w) + b  # 向量点积

    if tmp <= 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])

        """
        (0, 0) -> 0
        (1, 0) -> 0
        (0, 1) -> 0
        (1, 1) -> 1
        """
        print(str(xs) + " -> " + str(y))
