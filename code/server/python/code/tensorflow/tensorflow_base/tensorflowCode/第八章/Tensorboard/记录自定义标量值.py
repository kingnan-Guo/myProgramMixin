# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         记录自定义标量值
# Author:       Kingnan
# Date:         2020/6/14
# Description:

#   创建文件编写器 tf.summary.create_file_writer()

#   前期学习速率 非常的快LearningRateScheduler 控制学习速率，
#   # tf.keras.callbacks.LearningRateScheduler LearningRateScheduler 控制学习速率，
#   自定义学习速率 者将被传给kreas LearningRateScheduler 回调
#   tf.summary.scalar 记录自定义学习率      计算学习速率
#   将 LearningRateScheduler的回调 传给 Model.fit()
#
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

# 使用 手写数据集

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()

# 扩增 维度
train_image = tf.expand_dims(train_image, -1)
test_image = tf.expand_dims(test_image, -1)
# print('在最后一个 为度扩充 (60000, 28, 28, 1) ====', train_image[5:6])

# 改变数据类型
# train_image/255 归一化
train_image_float = tf.cast(train_image/255, tf.float32)
train_label_int = tf.cast(train_label, tf.int64)


test_image_float = tf.cast(test_image/255, tf.float32)
test_label_int = tf.cast(test_label, tf.int64)


# tf.data.Dataset.from_tensor_slices
# 该函数是dataset核心函数之一，它的作用是把给定的元组、列表和张量等数据进行特征切片。切片的范围是从最外层维度开始的。如果有多个特征进行组合，那么一次切片是把每个组合的最外维度的数据切开，分成一组一组的。
dataset = tf.data.Dataset.from_tensor_slices((train_image_float, train_label_int))

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_float, test_label_int))


print('dataset ==', dataset)
dataset_shuffle = dataset.shuffle(10000)
print('dataset_shuffle ==', dataset_shuffle)
dataset_batch = dataset_shuffle.batch(32)
print('dataset_batch ==', dataset_batch)

# 测试数据集
test_dataset_batch = test_dataset.batch(32)


model = tf.keras.Sequential([
    # tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(28, 28, 1))
    # None 任意形狀
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu', input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# model 的可训练参数
# print('model 的可训练参数  =>', model.trainable_variables)

optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=optimizer,
    loss=loss_func,
    metrics=['accuracy']
)

# tf.keras.callbacks.EarlyStopping EarlyStopping 若训练五个 epochs，  loss 依然没有下降则 停止训练
# tf.keras.callbacks.LearningRateScheduler LearningRateScheduler 控制学习速率，

# TensorBoard  参数 log_dir='logs' 要把执行文件 tensorboard 转成事件文件放在哪个路径下
# histogram_freq 记录直方图的频率 每一个epochs 记录一次 那就 设置为 1

log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


file_witer = tf.summary.create_file_writer(log_dir + './learnRead')
# 设置默认的 文件编写器
file_witer.set_as_default()




# 根据epochs 改变学习速率
def learnRead_sche(epoch):
    learning_rate = 0.2
    if epoch > 5:
        learning_rate = 0.02
    if epoch > 10:
        learning_rate = 0.01
    if epoch > 20:
        learning_rate = 0.005
    tf.summary.scalar('learning_rate', data=learning_rate, step=epoch)
    return learning_rate

# 需要一个参数（函数）  就是 learnRead_sche
learnRead_callback = tf.keras.callbacks.LearningRateScheduler(learnRead_sche)
# 这样就可以在 训练过程中 控制学习速率



model.fit(dataset_batch,
          epochs=10,
          steps_per_epoch=60000//128,
          validation_data=test_dataset_batch,
          validation_steps=10000//128,
          callbacks=[tensorboard_callback, learnRead_callback]
          )
