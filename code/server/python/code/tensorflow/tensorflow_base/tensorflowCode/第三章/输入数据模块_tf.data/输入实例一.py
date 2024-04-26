# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         输入实例一
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


(train_image, train_lable), (test_image, test_lable) = tf.keras.datasets.mnist.load_data()
print('train_image', train_image.shape)
# plt.imshow(train_image[0])
# plt.show()

# 因为是rgb 的颜色，所以 归一化除以255
# print('train_image / 255 \n', train_image/255)
train_image = train_image/255.0
test_image = test_image/255.0


# train_image.shape = (60000, 28, 28)
# 输入是一个 tensor，函数将样本个数识别为60000，然后对张量切片，每个样本ds_train_img的维度是（28, 28）
ds_train_img = tf.data.Dataset.from_tensor_slices(train_image)
print('ds_train_img', ds_train_img)

ds_train_lable = tf.data.Dataset.from_tensor_slices(train_lable)
print('ds_train_lable', ds_train_lable)
print(ds_train_lable)

# zip 将两个数组合并
# ds_train =  <ZipDataset shapes: ((28, 28), ()), types: (tf.float64, tf.uint8)>
ds_train = tf.data.Dataset.zip((ds_train_img, ds_train_lable))
print('ds_train = ', ds_train)

ds_train_shuffle = ds_train.shuffle(60000)
ds_train_shuffle_repeat = ds_train_shuffle.repeat()
ds_train_shuffle_repeat_batch = ds_train_shuffle_repeat.batch(64)
print('ds_train_shuffle', ds_train_shuffle_repeat_batch)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc'],
)

ds_test_img = tf.data.Dataset.from_tensor_slices(test_image)
ds_test_lable = tf.data.Dataset.from_tensor_slices(test_lable)
ds_test = tf.data.Dataset.zip((ds_test_img, ds_test_lable))
ds_test_batch = ds_test.batch(64)
# 每一个 epochs 需要训练多少步
#  // 整除
steps_per_epoch = train_image.shape[0]//64
print('steps_per_epoch 每一个 epochs 需要训练多少步', steps_per_epoch)
validation_steps = test_image.shape[0]//64
print('validation_steps 每一个测试数据集 训练多少步才把 所有测试数据验证完成', validation_steps)
model.fit(
    ds_train_shuffle_repeat_batch,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_test_batch,
    validation_steps=validation_steps
)

