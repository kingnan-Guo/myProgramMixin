# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         softmax多维分类_fashion_minist数据集
# Author:       Kingnan
# Date:         2020/5/30
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
(train_image, train_lable), (test_image, test_lable) = tf.keras.datasets.fashion_mnist.load_data()
# print(train_image.shape)
# plt.imshow(train_image[0])
# plt.show()
# print('train_lable', train_lable)

# 因为是rgb 的颜色，所以 归一化除以255
# print('train_image / 255 \n', train_image/255)
train_image = train_image/255.0
test_image = test_image/255.0

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
# 当 label 为顺序标签的时候 使用 sparse_categorical_crossentropy
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(train_image, train_lable, epochs=1)
model.evaluate(test_image, test_lable)
