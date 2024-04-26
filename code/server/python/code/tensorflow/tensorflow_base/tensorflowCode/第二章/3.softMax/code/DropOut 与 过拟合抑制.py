# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         DropOut 与 过拟合抑制
# Author:       Kingnan
# Date:         2020/5/31
# Description:  
#  添加  dropout 层 训练时随机丢弃部分 隐藏层的数值  tf.keras.layers.Dropout(0.5) 随丢弃 50%
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
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# tf.keras.layers.Dropout(0.5) 随丢弃 50%
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
# 当 label 读 热编码的时候 使用 categorical_crossentropy
# 需要将 label 改成 oneHot 编码的形式， 使用 tf.keras.utils.to_categorical()
tarin_label_oneHot = tf.keras.utils.to_categorical(train_lable)
# print('tarin_label_oneHot', tarin_label_oneHot[0])
test_label_oneHot = tf.keras.utils.to_categorical(test_lable)
# print('test_label_oneHot', test_label_oneHot[0])
model.compile(
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['acc']
)
# validation_data 在训练过程中就可以知道 训练的情况
history = model.fit(train_image, tarin_label_oneHot, epochs=5, validation_data=(test_image, test_label_oneHot))
# model.evaluate(test_image, test_label_oneHot)
# predict = model.predict(test_image)
# print('predict.shape ==', predict.shape)
# print('predict[0] == ', predict[0])
# print('np.argmax(predict[0])返回数组中 最大的 值的 key =', np.argmax(predict[0]))

print('history', history)
print('history.history', history.history)
print('history.history.key()', history.history.keys())

plt.plot(history.epoch, history.history.get('loss'), label='loss')
plt.plot(history.epoch, history.history.get('val_loss'), label='val_loss')



plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
plt.legend() # 将样例显示出来
plt.show()