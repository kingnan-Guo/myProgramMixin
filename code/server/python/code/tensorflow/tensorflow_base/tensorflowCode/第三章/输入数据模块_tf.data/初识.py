# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         初识
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
print(dataset)
for item in dataset:
    print('item', item)
    print('item.numpy()', item.numpy())

dataset2 = tf.data.Dataset.from_tensor_slices([[1, 2, 3],[2, 3, 4], [3, 4, 5,], [6, 7, 8]])
print(dataset2)
for item2 in dataset2:
    print('item2', item2.numpy())

# 使用字典创建
dataset_dic = tf.data.Dataset.from_tensor_slices({
    'a': [1, 2, 3, 4],
    'b': [3, 4, 5, 6],
    'c': [7, 8, 9, 0],
})
print(dataset_dic)
for item_dic in dataset_dic:
    print('item_dic', item_dic)