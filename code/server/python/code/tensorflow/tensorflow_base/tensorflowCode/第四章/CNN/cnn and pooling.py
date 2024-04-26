# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         cnn and pooling
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


# 1、代码中的DATA_FORMAT = 'channels_first'，即通道维靠前
# [batch，in_channels，in_height，in_with]既批次：批次大小渠道：通道数高度：图片的长宽度：宽
# 2、当然也可以和tensorflow中的数据格式一样指定 channels_last = 'channels_last',keras默认为channels_last
# tf.keras.layers.Conv2D()

(train_image, train_lable), (test_image, test_lable) = tf.keras.datasets.fashion_mnist.load_data()

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu', padding='same')
)
# model.output_shape (None, 26, 26, 32)
# print('model.output_shape', model.output_shape)
model.add(tf.keras.layers.MaxPool2D())
# print('model.output_shape pooling', model.output_shape)
model.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
)
print('model.output_shape Conv2D last', model.output_shape)
# 对前两个维度进行 平均值运算
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['acc']
)

train_image = train_image.reshape(-1, 28, 28, 1)
test_image = test_image.reshape(-1, 28, 28, 1)
print('train_image.shape', train_image.shape)
print('test_image.shape', test_image.shape)
history = model.fit(
    train_image,
    train_lable,
    epochs=30,
    validation_data=(test_image, test_lable)
)

plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')

plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend() # 将样例显示出来
plt.show()