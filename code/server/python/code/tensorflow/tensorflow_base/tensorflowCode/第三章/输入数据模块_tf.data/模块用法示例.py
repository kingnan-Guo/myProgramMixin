# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         模块用法示例
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

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8])
print(next(iter(dataset.take(1))))

# 数据乱序  shuffle

print('dataset.shuffle()', dataset.shuffle(8))
dataset_shuffle = dataset.shuffle(8)

for item in dataset_shuffle:
    print('item.numpy()', item.numpy())

# dataset_shuffle.repeat(2)  重复几次, 如果不填参数 count 则会一直重复下去
dataset_repeat = dataset_shuffle.repeat(count=2)
for item in dataset_repeat:
    print('item.numpy()', item.numpy())


# dataset_shuffle.batch(batch_size=2) 分批侧获取数据，dataset_batch[0].numpy() = [6 5]
dataset_batch = dataset_shuffle.batch(batch_size=5)
for item in dataset_batch:
    print('dataset_batch item.numpy()', item.numpy())

# 使用函数 对数据进行平方
dataset = dataset.map(tf.square)
for item in dataset:
    print('dataset_batch item.numpy()', item.numpy())