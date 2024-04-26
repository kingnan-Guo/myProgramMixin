# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         逻辑回归
# Author:       Kingnan
# Date:         2020/5/20
# Description:  返回的是和否
#  损失函数 ；sigmoid  y = 1/ (1 + e^(-x))  y = {0， 1}
# 交叉熵 损失函数 ： 实际输出的概率 与期望输出概率的 距离， 也就是交叉熵越小， 两个概率分布 越接近
#  2.多层感知器 是一个 映射网络， 映射到 从 0到1的值
#  关键词 binary_crossentropy 二元交叉熵
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.python.keras import Sequential
# 是否产生欺诈
csvPath = 'D:/Program Files/python/python_base/tensorflow2/tensorflow_base/resourceData/dateset/csv'
data = pd.read_csv(csvPath + '/credit-a.csv', header=None)
print(data.head())
print(data.iloc[:, -1].value_counts())
# print('data.iloc[:, -1]\n', data.iloc[:, -1])

x = data.iloc[:, :-1]
# 将15 列的 数字 -1 改成 0
y = data.iloc[:, -1].replace(-1, 0)
# print('x ===\n', x)
model = tf.keras.Sequential()

# Dense即全连接层，逻辑上等价于这样一个函数：
# 权重W为m*n的矩阵.
# 输入x为n维向量.
# 激活函数Activation.
# 偏置bias.
# 输出向量out为m维向量.
# out=Activation(Wx+bias).
# 即一个线性变化加一个非线性变化产生输出.
# Dense   units: Any, activation: Any = None
model.add(tf.keras.layers.Dense(units=6, input_shape=(15, ), activation='relu'))
model.add(tf.keras.layers.Dense(units=6, activation='relu'))
# 输出只有 0 1 所以一个神经元
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model.summary()
model.compile(
    optimizer='adam',
    # 二元交叉熵 binary_cross_entropy
    loss='binary_crossentropy',
    # metrics 计算正确率
    metrics=['acc']
)
history = model.fit(x, y, epochs=100)
print('history ===== \n', history)
print(history.history.keys())

plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('acc'))
plt.show()

input = data.iloc[:3, :-1]
print('input =============\n', input)
outPut = data.iloc[:3, -1]
print('outPut =============\n', outPut)
print('pred === \n', model.predict(input))
