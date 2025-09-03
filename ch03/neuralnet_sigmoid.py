
import numpy as np
from sigmoid import sigmoid

print("######### demo1.三层神经元的实现 #########")
print("######### demo1.1 从【输入层】到【第1层】的信号传递 #########")
# 从输入层到第1层的信号传递
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

"""
矩阵乘法运算，乘积也称为点积

X：1x2
W1：2x3
np.dot(X, W1)-》1x3
"""
# [0.2 0.5 0.8] + [0.1, 0.2, 0.3]
A1 = np.dot(X, W1) + B1
# 激活函数
Z1 = sigmoid(A1)
# [0.57444252 0.66818777 0.75026011]
print(Z1)

print("######### demo1.2 从【第1层】到【第2层】的信号传递 #########")
# 从第1层到第2层的信号传递
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
# [0.62624937 0.7710107 ]
print(Z2)

print("######### demo1.3 从【第2层】到【输出层】的信号传递 #########")

"""
输出层设计：恒等函数
输出层的激活函数
"""
def identity_function(x):
 return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
# [0.31682708 0.69627909]
print(Y)


print("######### demo2.三层神经元的实现代码小结 #########")
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
# [0.31682708 0.69627909]
print(y)
