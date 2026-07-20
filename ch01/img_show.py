# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

"""
从ch01目录下运行代码
"""
img = imread('../dataset/lena.png')  # 画像の読み込み
plt.imshow(img)  # 加载图片

plt.show()
