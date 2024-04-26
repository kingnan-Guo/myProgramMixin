# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         keras函数式 API 
# Author:       Kingnan
# Date:         2020/5/31
# Description:  
#   1、可以 实现 对网络中的 某一个层直接调用
#   2、 多输入的模型 中间使用  concatenate  进行合并
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(train_image, train_lable), (test_image, test_lable) = tf.keras.datasets.fashion_mnist.load_data()

# 因为是rgb 的颜色，所以 归一化除以255
# print('train_image / 255 \n', train_image/255)
train_image = train_image/255.0
test_image = test_image/255.0

# 要传入 网络的参数\
# input  = Tensor("input_1:0", shape=(None, 28, 28), dtype=float32)
input = tf.keras.Input(shape=(28, 28))
# conv1_Flatten(input) = Tensor("flatten/Identity:0", shape=(None, 784), dtype=float32)
conv1_Flatten = tf.keras.layers.Flatten()(input)

# conv2_Dense(input) = Tensor("dense/Identity:0", shape=(None, 28, 32), dtype=float32) 为什么有 28？ 因为
# conv2_Dense(conv1_Flatten) = Tensor("dense/Identity:0", shape=(None, 32), dtype=float32)
conv2_Dense = tf.keras.layers.Dense(32, activation='relu')(conv1_Flatten)
#  conv3_Dropout(conv2_Dense) = Tensor("dropout/Identity:0", shape=(None, 32), dtype=float32)
conv3_Dropout = tf.keras.layers.Dropout(0.5)(conv2_Dense)

conv4_Dense = tf.keras.layers.Dense(32, activation='relu')(conv3_Dropout)

output = tf.keras.layers.Dense(10, activation='softmax')(conv4_Dense)

# print('outPut', outPut)
# 传入模型的输入输出
model = tf.keras.Model(inputs=input, outputs=output)
model.summary()


# ------------
# 多输入的模型
input1 = tf.keras.Input(shape=(28, 28))
input2 = tf.keras.Input(shape=(28, 28))
conv_input1_Flatten = tf.keras.layers.Flatten()(input1)
conv_input2_Flatten = tf.keras.layers.Flatten()(input2)
# 合并 两个层 tf.keras.layers.concatenate
# conv1_concat = Tensor("concatenate/Identity:0", shape=(None, 1568), dtype=float32)
conv1_concat = tf.keras.layers.concatenate([conv_input1_Flatten, conv_input2_Flatten])

conv2_concat_Dense = tf.keras.layers.Dense(32, activation='relu')(conv1_concat)
output_concat = tf.keras.layers.Dense(1, activation='sigmoid')(conv2_concat_Dense)

model_concat = tf.keras.Model(inputs=[input1, input2], outputs=output_concat)
model_concat.summary()