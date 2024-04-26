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


# 变量
v = tf.Variable(0.0)
print((v + 1))
print((v + 1).numpy())

print(' v * 5 = ',v.assign(5))
print(' v + 1 = ',v.assign_add(1))
print(' 读取 tensor ',v.read_value())
# print('保存 变量的 值', )

# 自动微分
w = tf.Variable([[2.0]])

# 求解梯度
# 梯度磁带 记录运算过程
# 自动跟踪 变量的 运算
with tf.GradientTape() as t:
    loss = w * w

# loss 相对于 w 的 微分是多少   loss(x)' ==
# loss : f(x) = x^2   ,求 导数 f(x)' = ？  ； 结果为 2x
grad = t.gradient(loss, w)
print("因为w设置变量为 2.0，当变量w = 2的时候, f(x)'= 2x ; 带入 x = 2 , 最终  grad =", grad)

# 如果设置常量

b = tf.constant(3.0)
with tf.GradientTape() as t:
    # 因为 b 为常量， 所以添加 t.watch; 让t跟踪常量的运算
    t.watch(b)
    b_loss = b * b

b_grad = t.gradient(b_loss, b)
#  b为常量 ，求导后为 0 ,但是结果与我理解的不相符
print('; b_grad =', b_grad)






#   在计算过程中  调用 t.gradient 会立即释放资源, 若想做多个微分,那么要添加传值
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    # 因为 b 为常量， 所以添加 t.watch; 让t跟踪常量的运算
    t.watch(x)
    y = x * x
    z = y * y

dy_dx = t.gradient(y, x)
#  x^2 求导后 为 2想，带入 常数 3 等于 6
print('dy_dx =', dy_dx)

# print('在 tf.GradientTape() 没有参数的时候 ，再次进行微分运算，会报错 RuntimeError， kareas 如果没有 遍历的话也会报这个错； dz_dw = ', dz_dw)

#  persistent=True , 会永久记录  梯度磁带 内的运算过程
# 就可以多次进行微分运算了

# z = y * y ； y= x*x  z对x的导数，
dz_dw = t.gradient(z, x)
print('dz_dw = ', dz_dw)

dz_dy = t.gradient(z, y)
print('dz_dy = ', dz_dy)




# 使用 手写数据集

(train_image, train_label), (test_iamge, test_label) = tf.keras.datasets.mnist.load_data()
print(train_image.shape)
# print(train_image[5:6])
# 扩增 维度
train_image = tf.expand_dims(train_image, -1)
# print('在最后一个 为度扩充 (60000, 28, 28, 1) ====', train_image[5:6])

# 改变数据类型
# train_image/255 归一化
train_image_float = tf.cast(train_image/255, tf.float32)
train_label_int = tf.cast(train_label, tf.int64)
# tf.data.Dataset.from_tensor_slices
# 该函数是dataset核心函数之一，它的作用是把给定的元组、列表和张量等数据进行特征切片。切片的范围是从最外层维度开始的。如果有多个特征进行组合，那么一次切片是把每个组合的最外维度的数据切开，分成一组一组的。
dataset = tf.data.Dataset.from_tensor_slices((train_image_float, train_label_int))

print('dataset ==', dataset)
dataset_shuffle = dataset.shuffle(10000)
print('dataset_shuffle ==', dataset_shuffle)
dataset_batch = dataset_shuffle.batch(32)
print('dataset_batch ==', dataset_batch)


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



# 做一步数据的 训练

def train_step(model, images, labels):
    # 计算损失值 与我们 可训练参数之间的 梯度
    with tf.GradientTape() as t:
        loss_step = loss(model, images, labels)
    # 计算 loss_sstep 与 model 的可训练参数 的梯度
    grad = t.gradient(loss_step, model.trainable_variables)

    # 使梯度下降最快， 使用 优化器
    # 应用 apply_gradients 这个该方法 改变 变量值 使梯度下降最快
    # 会改变 model.trainable_variables 的值 向梯度下降最快的方向
    optimizer.apply_gradients(zip(grad, model.trainable_variables))



def trian ():
    # 训练多少个 epoch
    for epoch in range(10):
        for (batch, (images, labels)) in enumerate(dataset_batch):
            # 进行训练
            # print('batch ===', batch, '(images, labels) == ', (images, labels))
            train_step(model, images, labels)
        print('Epoch {} is finshed'.format(epoch))

trian()