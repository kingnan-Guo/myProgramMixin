# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         调整 adam 优化器的参数
# Author:       Kingnan
# Date:         2020/5/31
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.python.keras import Sequential






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
# 当 label 读 热编码的时候 使用 categorical_crossentropy
# 需要将 label 改成 oneHot 编码的形式， 使用 tf.keras.utils.to_categorical()
tarin_label_oneHot = tf.keras.utils.to_categorical(train_lable)
# print('tarin_label_oneHot', tarin_label_oneHot[0])
test_label_oneHot = tf.keras.utils.to_categorical(test_lable)
# print('test_label_oneHot', test_label_oneHot[0])
# tf.keras.optimizers.Adam() 优化器， learning_rate学习速率
model.compile(
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['acc']
)

model.fit(train_image, tarin_label_oneHot, epochs=5)
# model.evaluate(test_image, test_label_oneHot)
predict = model.predict(test_image)
print('predict.shape ==', predict.shape)
print('predict[0] == ', predict[0])
print('np.argmax(predict[0])返回数组中 最大的 值的 key =', np.argmax(predict[0]))