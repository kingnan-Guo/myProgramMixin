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



# 使用 手写数据集

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
print(train_image.shape)
# print(train_image[5:6])
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
    tf.keras.layers.Dense(10),
])

# model 的可训练参数
print('model 的可训练参数  =>', model.trainable_variables)

optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# model.compile(
#     optimizer='adam',
#     loss=
# )


# iter() 函数用来生成迭代器。
print('iter(dataset) ==>', iter(dataset))
features, labels = next(iter(dataset_batch))
print('feature', features.shape)
prediction = model(features)
print('prediction', prediction)

# 找到 prediction 中最大的值

prediction_argmax = tf.argmax(prediction, axis=1)
print('prediction_argmax', prediction_argmax)

print('labels ==', labels)

# 训练过程
def loss(model, x, y):
    y_pred = model(x)
    # 计算交叉熵损失
    return  loss_func(y_true=y, y_pred=y_pred)


#  train_loss 命名
train_loss = tf.keras.metrics.Mean(name='train_loss')
# 计算训练过程中的 平均正确率
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


#  test_loss 命名
test_loss = tf.keras.metrics.Mean(name='test_loss')
# 计算训练过程中的 平均正确率
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# 做一步数据的 训练

def train_step(model, images, labels):
    # 计算损失值 与我们 可训练参数之间的 梯度
    with tf.GradientTape() as t:
        # pred 得到一个预测结果
        pred = model(images)
        # loss_step 每一步的预测值
        loss_step = loss_func(labels, pred)

        # loss_step = loss(model, images, labels)
    # 计算 loss_sstep 与 model 的可训练参数 的梯度
    grad = t.gradient(loss_step, model.trainable_variables)

    # 使梯度下降最快， 使用 优化器
    # 应用 apply_gradients 这个该方法 改变 变量值 使梯度下降最快
    # 会改变 model.trainable_variables 的值 向梯度下降最快的方向
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
    # train_loss(loss_step) 获取 loss 的均值
    train_loss(loss_step)
    # 平均正确率情况
    train_accuracy(labels, pred)



def test_step(model, images, labels):
    # 没有 梯度优化
    # pred 得到一个预测结果 test_image
    pred = model(images)
    # loss_step 每一步的预测值
    loss_step = loss_func(labels, pred)
    # train_loss(loss_step) 获取 loss 的均值
    test_loss(loss_step)
    # 平均正确率情况
    test_accuracy(labels, pred)



def trian ():
    # 训练多少个 epoch
    for epoch in range(10):
        for (batch, (images, labels)) in enumerate(dataset_batch):
            # 进行训练
            # print('batch ===', batch, '(images, labels) == ', (images, labels))
            train_step(model, images, labels)

        for (test_batch, (test_images, test_labels)) in enumerate(test_dataset_batch):
            # 进行训练
            # print('batch ===', batch, '(images, labels) == ', (images, labels))
            test_step(model, test_images, test_labels)

        print('Epoch {} Loss is {}, accuracy is {} '.format(epoch, train_loss.result(), train_accuracy.result()))
        print('Epoch {} test_Loss is {}, test_accuracy is {} '.format(epoch, test_loss.result(), test_accuracy.result()))

        # 状态 重置 每个 epoch 结束后 都会重置
        train_loss.reset_states()
        train_accuracy.reset_states()

        test_loss.reset_states()
        test_accuracy.reset_states()
trian()



