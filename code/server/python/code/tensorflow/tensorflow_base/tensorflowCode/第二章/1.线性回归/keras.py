# _*_　 coding: utf-8 _*_
__author__ = "kingnan"
__date__ = '2020/5/17 18:56'

import tensorflow as tf
import pandas as pd
import numpy as np
import os
print('TEnsorflow Version: {} '.format(tf.__version__))  #输出版本号

# path=os.path.abspath('.....')  # 表示当前所处的文件夹上一级文件夹的绝对路径
# print(path + '/resourceData/dateset/csv/Income1.csv')

# print('os', os.path.realpath('D:/Program Files/python/python_base/tensorflow2/tensorflow_base/resourceData/dataSet/csv'))
# data = pd.read_csv('D:/Program Files/python/python_base/tensorflow2/tensorflow_base/resourceData/dateset/csv/Income1.csv')


data = pd.read_csv('/Users/kingnan/program/code/server/python/code/tensorflow/tensorflow_base/resourceData/dateset/csv/Income1.csv')
print(data)

import matplotlib.pyplot as plt
# %matplotlib inline
plt.scatter( x=data.Education, y=data.Income)
# plt.show()


# 均方差  (f(x) - y)^2  预测的值与 真实的值的均方差越小越好
# f(x) = ax + b
model = tf.keras.Sequential()
# tf.keras.layers.Dense(输出维度， 输入维度（元组）)
#
model.add(tf.keras.layers.Dense(1, input_shape=(1, )))
model.summary()

# 使用梯度下降算法
model.compile(
    # 优化方法
    optimizer='adam',
    # mse 均方差
    loss='mse'
)
x=data.Education
y=data.Income
# epochs 对所有数据的 训练次数
# history 记录训练过程
history = model.fit(x, y, epochs=5000)
# print(history)
# 预测结果
print(model.predict(x))


print(model.predict(np.array([20])))