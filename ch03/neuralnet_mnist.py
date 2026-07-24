# coding: utf-8
import os
import sys

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from common.timer import timer


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """
    进行分类，softmax 概率
    :param network:
    :param x:
    :return:
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


with timer("神经网络推理for循环单个处理"):
    """
    加载数据
    
    x 是 10000 行（测试数据样本） x 784 列（28 x 28 像素，展平）
    t 是 10000 行（标签）
    """
    x, t = get_data()
    """
    
    加载训练好的神经网络
     +-----------------+                +-----------------+                +-----------------+                +-----------------+
    
     |     输入层      |  W1: (784, 50) |    隐藏层 1        |  W2: (50, 100) |    隐藏层 2     |  W3: (100, 10) |     输出层      |
     |   784 个节点    | -------------> |    50 个节点       | -------------> |   100 个节点    | -------------> |    10 个节点    |
     |                 |   b1: (50,)    |                 |   b2: (100,)   |                 |   b3: (10,)    |                 |
     +-----------------+                +-----------------+                +-----------------+                +-----------------+
     
     说明：
     我们对这个MNIST数据集实现神经网络的推理处理。神经网络的输入层有784个神经元，输出层有10个神经元。
     - 输入层的784这个数字来源于图像大小的28 × 28 = 784，
     - 输出层的10这个数字来源于10类别分类（数字0到9，共10类别）
     此外，这个神经网络有2个隐藏层，第1个隐藏层有50个神经元，第2个隐藏层有100个神经元。这个50和100可以设置为任何值
    """
    network = init_network()

    accuracy_cnt = 0  # 精度计数器，识别正确累计
    for i in range(len(x)):  # 10000
        # 根据测试数据，得到预测值，np.sum(y)=1
        """
        [8.4412648e-05 2.6350654e-06 7.1549491e-04 1.2586274e-03 1.1727966e-06
        4.4990895e-05 1.6269318e-08 9.9706501e-01 9.3744884e-06 8.1831194e-04]
        """
        y = predict(network, x[i])
        # 获取最大概率的索引，返回 0-9
        p = np.argmax(y)  # 最も確率の高い要素のインデックスを取得
        # 预测值和真实标签（标签值也是0-9）进行比较，如果相同则计数加1
        if p == t[i]:
            accuracy_cnt += 1

    # Accuracy:0.9352
    # 这表示有 93.52 %的数据被正确分类了
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
