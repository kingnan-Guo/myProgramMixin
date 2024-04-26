# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         数据读取
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



# 路径管理
import pathlib
import random
BATCH_SIZE = 32

data_dir = '/Users/kingnan/program/code/server/python/code/tensorflow/tensorflow_base/resourceData/dateset/dog and cat/dc_2000'
# data_dir = '../../../../../python/code/tensorflow/tensorflow_base/resourceData/dateset/dog and cat/dc_2000'

data_root = pathlib.Path(data_dir)
print(data_root)

all_train_image_path = data_root.glob('./train/*/*.jpg')
all_train_image_path_list = list(all_train_image_path)
print('all_train_image_path_list', len(all_train_image_path_list))
all_train_image_path_str_list = [str(path) for path in all_train_image_path_list]
print('all_train_image_path_str_list', all_train_image_path_str_list[-5])

train_image_label = [int(P.split('/')[15] == 'cat') for P in all_train_image_path_str_list]
print('train_image_label ==', train_image_label)

# 合并 label  url
def load_preprosess_image(path, label):
    image = tf.io.read_file(path)
    print('load_preprosess_image image ==', image)
    image = tf.image.decode_jpeg(image, channels=3)
    print('load_preprosess_image image ==', image)
    # 整理图像大小
    image_resize = tf.image.resize(image, [256, 256])
    print('image_resize', image_resize)
    image_float = tf.cast(image_resize, tf.float32)
    print('image_float', image_float)
    # 归一化
    image_tensor_normalize = image_float / 255
    # print('image_tensor_normalize', image_tensor_normalize)
    return image_tensor_normalize, label



# load_preprosess_image(all_train_image_path_str_list[1], 1)

# 创建 dataset
# tf.data.Dataset 数据仓库的形式
# 同时使用image_label 和image_dataset
train_image_dataset = tf.data.Dataset.from_tensor_slices((all_train_image_path_str_list, train_image_label))

# 根据计算机cpu的个数，自动使用并行内存
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_image_dataset_map = train_image_dataset.map(load_preprosess_image, num_parallel_calls=AUTOTUNE)

print('train_image_dataset_map =', train_image_dataset_map)


# 设置 batch_size  乱序
batch_size = 32
train_count = len(all_train_image_path_list)
train_image_dataset_map_shuffle_batch_size = train_image_dataset_map.shuffle(train_count).batch(batch_size)


# prefetch预读取数据，在读取数据的过程中会预先读取一部分数据，加快运行速度
# AUTOTUNE 让后台自己做决定
train_image_dataset_map_shuffle_batch_prefetch = train_image_dataset_map_shuffle_batch_size.prefetch(buffer_size=AUTOTUNE)

# 使用 iter 把数据转为可迭代的数据
# 使用 next 迭代 一步
image, labels = next(iter(train_image_dataset_map_shuffle_batch_prefetch))


print('image ', image.shape, 'label = ', labels.shape)

# plt.imshow(image[0])
# plt.show()

# 创建简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), activation='relu'),
    # GlobalAveragePooling2D 在第二第三维度上同时求平均
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1),
])


model.summary()
# pred = model(image)
# print(pred.shape)
# print('---', np.array([p[0].numpy() for p in tf.cast(pred > 0, tf.int32)]))
# print('---', np.array([l.numpy() for l in labels]))

# 损失函数
# 二元交叉熵
# BinaryCrossentropy 返回可调用对象
loss = tf.keras.losses.BinaryCrossentropy()
print('loss', loss)
# 计算这两个部分的交叉熵损失函数

print('loss ==', loss([0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]))

print('loss ==', loss([[0.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]]))

loss2 = tf.keras.losses.binary_crossentropy([0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0])
print('loss ---2', loss2)

optimizer = tf.keras.optimizers.Adam()

# 每一个epoch loss 的平均值
epoch_loss_avg = tf.keras.metrics.Mean('train_loss')
# 正确率
train_accuracy = tf.keras.metrics.Accuracy()


def train_step(model, images, labels):
    # 记录计算的梯度
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, pred)
    # 计算损失值与 可训练参数 之间的梯度
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss_avg(loss_step)
    print(pred)
    # 计算预测值之间的准确率
    train_accuracy(labels, tf.cast(pred>0, tf.int32))











# 定义 训练
train_loss_result = []
train_acc_result = []
num_epochs = 1
for epoch in range(num_epochs):
    print(epoch)
    for _imgs, _labels in train_image_dataset_map_shuffle_batch_prefetch:
        train_step(model, _imgs, _labels)
        print('.', end='')
    print()
    #
    train_loss_result.append(epoch_loss_avg.result())
    train_acc_result.append(train_accuracy.result())

    print('Epoch: {};loss: {:.3f}, accuracy: {:3f}'.format(epoch + 1, epoch_loss_avg.result(), train_accuracy.result()))
    # 若没有重置状态，那么就 会计算所有数据的平均值
    epoch_loss_avg.reset_states()
    train_accuracy.reset_states()



# pred = [1,0,0,1]
# print('----==',tf.cast(1>0, tf.int32)
# print('train_accuracy', train_accuracy([1,1,0,1], [1,0,0,1])  )