# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         自动微分运算
# Author:       Kingnan
# Date:         2020/6/9
# Description:  
#   GradientTape 对于变量或者常量的运算 都属于float 类型
#   在计算过程中  调用 t.gradient 会立即释放资源, 若想做多个微分,那么要添加传值
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
model.fit(dataset_batch,
          epochs=5,
          steps_per_epoch=60000//128,
          validation_data=test_dataset_batch,
          validation_steps=10000//128,
          callbacks=[tensorboard_callback]
          )



# tensorboard --logdir=tensorboard --logdir=20200614-013847