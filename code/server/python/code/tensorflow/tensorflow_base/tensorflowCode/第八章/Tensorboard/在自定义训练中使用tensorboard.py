# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         在自定义训练中使用tensorboard
# Author:       Kingnan
# Date:         2020/6/14
# Description:  
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

optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss (model, x, y):
    y_ = model(x)
    return loss_func(y, y_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
#
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')

test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

def train_step(model, image, labels):
    with tf.GradientTape() as t:
        pred = model(image)
        # t.gradient()
        loss_step = loss_func(labels, pred)
    grad = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    train_loss(loss_step)
    train_accuracy(labels, pred)

def test_step(model, image, labels):
    pred = model(image)
    test_loss_step = loss_func(labels, pred)
    test_loss(test_loss_step)
    test_accuracy(labels, pred)


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
# train的数据储存路径
train_log_dir = 'logs/gradient_tape' + current_time + '/train'
test_log_dir = 'logs/gradient_tape' + current_time + '/test'

# 创建 文件编写器
train_weiter = tf.summary.create_file_writer(logdir=train_log_dir)
test_weiter = tf.summary.create_file_writer(logdir=test_log_dir)

def train():
    for epoch in range(10):
        for (batch, (image, labels)) in enumerate(dataset_batch):
            train_step(model, image, labels)

        with train_weiter.as_default():
            # 收集标量值 使用 train_weiter写再在磁盘上
            tf.summary.scalar(name='loss', data=train_loss.result(), step=epoch)
            tf.summary.scalar(name='acc', data=train_accuracy.result(), step=epoch)

        for (batch, (image, labels)) in enumerate(test_dataset_batch):
            test_step(model, image, labels)

        with test_weiter.as_default():
            # 收集标量值 使用 train_weiter写再在磁盘上
            tf.summary.scalar(name='test_loss', data=test_loss.result(), step=epoch)
            tf.summary.scalar(name='test_acc', data=test_accuracy.result(), step=epoch)


        template = 'Epoch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accracy: {}'
        print(
            template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100)
        )
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

train()