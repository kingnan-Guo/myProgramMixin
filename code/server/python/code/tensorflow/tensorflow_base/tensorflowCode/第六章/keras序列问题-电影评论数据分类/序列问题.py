# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         序列问题
# Author:       Kingnan
# Date:         2020/6/7
# Description:  
#
#
#-------------------------------------------------------------------------------

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------
#     # 导入电影评论数据
#     data = keras.datasets.imdb
#     # 加载电影评论数据
#     # load_data (num_words) 最大长度
#     (x_train, y_train), (x_test, y_test) = data.load_data(num_words=max_word)
# ------------------------------------------


# num_words=10000的意思是训练集中我们指保留词频最高的前10000个单词
# 10000名之后的词汇都会被直接忽略，不出现在train_data和test_data中
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

print(y_train)

#
# for x in y_train:
#     print(x)


# 将所有的数据长度 的填充 到300
x_train_300 = tf.keras.preprocessing.sequence.pad_sequences(x_train, 300)
x_test_300 = tf.keras.preprocessing.sequence.pad_sequences(x_test, 300)


test = 'i love you'
test.split()
# test.split().index(word) 获取 每一个词的索引
for word in test.split():
    print(test.split().index(word))

# dict((word, test.split().index(word)) for word in test.split())

model = keras.Sequential()
# 经过 Embedding 后 数据 （2500， 300）会变成 （2500， 300， 50）
# tf.keras.layers.Embbedding()只能作为模型第一层使用。
model.add(layers.Embedding(10000, 50, input_length=300))
# 经过 Flatten 后 会从 （2500， 300， 50） -> （2500， 15000 ）
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    # 二分类问题
    loss='binary_crossentropy',
    metrics=['acc']
)
model.fit(x_train_300, y_train, epochs=15, batch_size=256, validation_data=(x_test_300, y_test))
