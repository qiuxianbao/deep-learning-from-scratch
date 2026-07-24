# coding: utf-8
"""
mnist数据集的使用范例
1.加载
2.显示
"""

import os
import sys

"""
导入父目录，目的是为了能找到dataset这个模块

否则会报如下错误：
ModuleNotFoundError: No module named 'dataset'
"""
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    """
    PIL（Python Image Library）模块
    :param img:
    :return:
    """
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 加载mnist数据集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# print(x_train.shape)  # (60000, 784)
# print(t_train.shape)  # (60000,)

# print(x_test.shape)  # (10000, 784)
# print(t_test.shape)  # (10000,)

img = x_train[0]
# print(img.shape)  # (784,)

label = t_train[0]
# print(label)  # 5

test_img = x_test[0]
# print(test_img.shape)  # (784,)

test_label = t_test[0]
# print(test_label)  #  7

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)  # 5
