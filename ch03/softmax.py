import numpy as np


def softmax_by_steps():
    a = np.array([0.3, 2.9, 4.0])
    exp_a = np.exp(a)
    print(exp_a)

    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)

    y = exp_a / sum_exp_a
    print(y)


def softmax0(a):
    """
    激活函数

    缺陷：指数函数，存在溢出问题

    :param a:
    :return:
    """
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def test_softmax0():
    a = np.array([0.3, 2.9, 4.0])
    print(softmax0(a))


def test_softmax():
    # 值过大会导致溢出
    a = np.array([1010, 1000, 990])
    """
    RuntimeWarning: overflow encountered in exp
    """
    print(softmax0(a))  # [nan nan nan], nan（not a number，不确定）

    y = softmax(a)
    print(y)
    print(np.sum(y))


def _main():
    # softmax_by_steps()
    # test_softmax0()
    test_softmax()


if __name__ == '__main__':
    _main()
