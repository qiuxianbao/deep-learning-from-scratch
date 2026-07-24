# coding: utf-8
import sys, os

from common.timer import timer

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


with timer("神经网络推理for循环批量处理"):
    x, t = get_data()
    network = init_network()

    """
    与其一张一张地把图片送入网络计算（循环 10,000 次），不如一次性把 100 张图片打包成一个大矩阵丢进去
    
    """
    batch_size = 100  # 每次处理个数
    accuracy_cnt = 0

    """
   【 单批次 (Batch) 数据矩阵流向与维度转化图 】
                                   
     +-----------------------+
    
     |        x_batch        |  矩阵形状: (100, 784)
     | --------------------- |  [100张图片, 每张图片展平成784个像素点]
     | [图1: 784个像素特征]  |
     | [图2: 784个像素特征]  |
     |         ...           |
     +-----------------------+
                 |
                 |  传递给: predict(network, x_batch) 
                 |  内部执行 3 层全连接层矩阵相乘: 
                 |  X·W1 -> Layer1 · W2 -> Layer2 · W3
                 ▼
     +-----------------------+
    
     |        y_batch        |  矩阵形状: (100, 10)
     | --------------------- |  [100张图片在 0~9 这10个分类上的预测得分/概率]
     | [图1: 0.1, 0.8, 0.1...] |  --> 比如在索引1的位置得分最高
     | [图2: 0.0, 0.1, 0.9...] |  --> 比如在索引2的位置得分最高
     |         ...           |
     +-----------------------+
                 |
                 |  执行: np.argmax(y_batch, axis=1)
                 |  作用: 横向(按行)扫描，挑出每行最大值所在的下标索引
                 ▼
     +-----------------------+
    
     |           p           |  矩阵形状: (100,)
     | --------------------- |  [模型最终猜出来的 100 个数字答案]
     |   [ 1, 2, ..., 9 ]    |
     +-----------------------+
                 |
                 |             +-----------------------+
    
                 |             | t[i : i + batch_size] | 一维数组: (100,)
                 |             | --------------------- | [对应的 100 张图的真实标准答案]
                 |             |   [ 1, 3, ..., 9 ]    |
                 |             +-----------------------+
    
                 |                         |
                 +------------+------------+
                              |
                              | 执行: p == t[...] 逐元素比对
                              ▼
     +-----------------------+
    
     |       布尔矩阵          |  一维数组: (100,)
     | --------------------- |  [相同位置返回 True, 不同返回 False]
     | [True, False,..,True] |
     +-----------------------+
                 |
                 | 执行: np.sum(...)
                 | 作用: 隐式转换 (True->1, False->0) 并求和
                 ▼
         [ 算出本批猜对的张数 ] ───( += 累加到总量 )───>  accuracy_cnt

    """

    for i in range(0, len(x), batch_size):  # step是 batch_size 个
        # 当 i = 0 时，切出第 0 到 99 张图片；
        x_batch = x[i:i + batch_size]
        # y_batch 是一个形状为 (100, 10) 的二维概率矩阵
        y_batch = predict(network, x_batch)
        """
        p 是一个形状为 (100,) 的一维数组
        axis=1（横向降维）把二维的(100, 10) 压缩成了一维的(100, )从而可以和真实标签直接进行对比
        """
        p = np.argmax(y_batch, axis=1)

        """
        t[i:i + batch_size]：切片切出这 100 张图片对应的真实标准答案
        p == t[..]: 将模型的预测答案数组与真实答案数组进行逐元素对比，猜对是True，猜错是False
        np.sum(...): NumPy 中进行数学求和时，True 会被当成 1，False 会被当成 0
        """
        accuracy_cnt += np.sum(p == t[i:i + batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
